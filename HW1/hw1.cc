#include <cstdlib>
#include <iostream>
#include <mpi.h>
#include <boost/sort/spreadsort/spreadsort.hpp>

//buf data都是排序好的數字 tmp是要回傳的指標 
float* Merge_small(float* buf, float* data, float* tmp, int bufsize, int datasize){ //bufsize是要蒐集前幾小的數字 
    int s = 0; int i = 0; int j = 0; //index
    
    //如果一邊空了就取另一邊 i最多到bufsize-1, j最多到datasize-1
    while(s<bufsize){
        if(buf[i]<=data[j]){
            tmp[s] = buf[i];
            ++i;
            ++s;
            if(i==bufsize) break;
        }
        else{
            tmp[s] = data[j];
            ++j;
            ++s;
            if(j==datasize) break;
        }
    }

    if(s!=bufsize && i == bufsize){
        while(s<bufsize){
            tmp[s] = data[j];
            ++s;
            ++j;
        }
    }
    else if(s!=bufsize && j == bufsize){
        while(s<bufsize){
            tmp[s] = buf[i];
            ++s;
            ++i;
        }
    }
    return tmp;
}
float* Merge_large(float* buf, float* data, float* tmp, int bufsize, int datasize){ //bufsize是要蒐集前幾大的數字 
    int s = bufsize-1; 
    int i = bufsize-1;
    int j = datasize-1;
    
    //從後往前取大的
    //如果一邊空了就取另一邊 i最多到bufsize-1, j最多到datasize-1   
    while(s>=0){
        if(buf[i]>data[j]){
            tmp[s] = buf[i];
            --i;
            --s;
            if(i<0) break;
        }
        else{
            tmp[s] = data[j];
            --j;
            --s;
            if(j<0) break;
        }
    }

    if(s!=0 && i < 0){
        while(s>=0){
            tmp[s] = data[j];
            --s;
            --j;
        }
    }
    else if(s!=bufsize && j < 0){
        while(s>=0){
            tmp[s] = buf[i];
            --s;
            --i;
        }
    }
    return tmp;
    
}

int main(int argc, char **argv)
{
    int rc, rank, size, x, phase, nb, num_p, num_fp;
    int* bufsize = &x;
    float* buf, *data, *tmp, *ch;
    int * p = &num_p;
    int *fp = &num_fp;
    int y = 0;
    int *flag = &y; //special case =1
    int * nproc = &size;

    MPI_File input_file, output_file;
    MPI_Offset offset;

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n = atoi(argv[1]); //inputfile有多少數字
    char *input_filename = argv[2];
    char *output_filename = argv[3];
    
    if(n>=size){
        *p = n/size; //除了最後一個process外，每個process要負責的數字量
        *fp = n-(*p)*(size-1); //最後一個process要負責的數字量
    }
    else{  //數字量比process數目少 
        *p = 1;
        *fp = 1;
        *flag = 1; //special case
        *nproc = n;  //表示在這個case中要使用的process數量
        if(rank>n-1){
            *p = 0;
            *fp = 0;
        }
    }

    //配置每個process的buffer大小
    if(rank!=(*nproc-1)){
        *bufsize = *p;
    }
    else{
        *bufsize = *fp;
    }
    buf = new float[*bufsize];

    //建立交換資料用的陣列
    int datasize = *p > *fp? *p : *fp;   //大小以process中負責的最大資料量來建立
    tmp = new float[*bufsize]; //放merge好要放回自己buffer的資料
    data = new float[datasize]; //放鄰居的資料
    
    //計算每個process讀檔案的偏移量
    offset = rank * (*p) * sizeof(float);
    
 
    MPI_File_open(MPI_COMM_WORLD, input_filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &input_file);

    if(size!=1 && (*flag)!=1){  //一般情況
    MPI_File_set_view(input_file, offset, MPI_FLOAT, MPI_FLOAT, "native", MPI_INFO_NULL);
    MPI_File_read_all(input_file, buf, *bufsize, MPI_FLOAT, MPI_STATUS_IGNORE);
    }
    else if(size!=1 && (*flag)==1 && rank<n){  //n<nproc的特殊情況
        MPI_File_read_at(input_file, offset, buf, *bufsize, MPI_FLOAT, MPI_STATUS_IGNORE);
    }
    else if(size==1){ //n=1
    MPI_File_read(input_file, buf, *bufsize, MPI_FLOAT, MPI_STATUS_IGNORE);
    }

    MPI_File_close(&input_file);
    
    //各自sort自己array內的數字
    boost::sort::spreadsort::spreadsort(buf, buf + *bufsize);

    //配對odd-phase跟even-phase鄰居
    int o_nb, e_nb;
    int* odd_nb = &o_nb;
    int* even_nb = &e_nb;
    if(rank%2==0){
        *even_nb = rank+1;
        *odd_nb = rank-1;
        if(*even_nb == *nproc) *even_nb=-1;
    }
    else{
        *even_nb = rank-1;
        *odd_nb = rank+1;
        if(*odd_nb == *nproc) *odd_nb=-1;
    }

    if(*flag == 1 && rank>=n){ //每一回合都idle
        *even_nb = -1;
        *odd_nb = -1;
    }

    //取得鄰居數字量
    int odd_nbsz = *p;
    int even_nbsz = *p;
    int *odd_nbsize = &odd_nbsz;
    int *even_nbsize = &even_nbsz;
    if (*odd_nb == (*nproc-1)) *odd_nbsize = *fp;
    if (*even_nb == (*nproc-1)) *even_nbsize = *fp;

    //進入iteration
    //要注意最後一個process的buf長度可能比倒數第二個長或短
    //只有一個process就不用交換 直接輸出排好的結果
    if(size!=1){
        int phase = 0;
        while(phase<=*nproc){  
            //even-phase
            //把自己的資料傳給對方 左邊資料要有較小的數字
            if(*even_nb>=0){
                MPI_Sendrecv(buf, *bufsize, MPI_FLOAT, *even_nb, 0, data, *even_nbsize, MPI_FLOAT, *even_nb,  0, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
            }
            if(rank%2==0 && *even_nb>=0){
                tmp = Merge_small(buf, data, tmp, *bufsize, *even_nbsize);
                //此時tmp指向換好的較小數字 
                //對調buf跟tmp指標
                ch = buf;
                buf = tmp;
                tmp = ch;
            }
            else if(rank%2!=0 && *even_nb>=0){ 
                tmp = Merge_large(buf, data, tmp, *bufsize, *even_nbsize); 
                //此時tmp指向換好的較大數字 
                //對調buf跟tmp指標
                ch = buf;
                buf = tmp;
                tmp = ch;
            }
            
            //odd-phase
            if(*odd_nb>=0){
                MPI_Sendrecv(buf, *bufsize, MPI_FLOAT, *odd_nb, 0, data, *odd_nbsize, MPI_FLOAT, *odd_nb,  0, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
            }
            if(rank%2!=0 && *odd_nb>=0){
                tmp = Merge_small(buf, data, tmp, *bufsize, *odd_nbsize);
                //對調buf跟tmp指標
                ch = buf;
                buf = tmp;
                tmp = ch;
            }
            else if(rank%2==0 && *odd_nb>=0){ 
                tmp = Merge_large(buf, data, tmp, *bufsize, *odd_nbsize);
                //對調buf跟tmp指標
                ch = buf;
                buf = tmp;
                tmp = ch;
            }
            //odd phase 結束
            phase+=2;
        }
    }
   
    MPI_File_open(MPI_COMM_WORLD, output_filename, MPI_MODE_CREATE|MPI_MODE_WRONLY, MPI_INFO_NULL, &output_file);
    
    if(size!=1 && (*flag)==0){
        MPI_File_set_view(output_file, offset, MPI_FLOAT, MPI_FLOAT, "native", MPI_INFO_NULL);
        MPI_File_write_all(output_file, buf, *bufsize, MPI_FLOAT, MPI_STATUS_IGNORE);
    }
    else if(size!=1 && (*flag)==1 && rank<n){
        MPI_File_write_at(output_file, offset, buf, *bufsize, MPI_FLOAT, MPI_STATUS_IGNORE);
    }
    else if(size==1){
        MPI_File_write(output_file, buf, *bufsize, MPI_FLOAT, MPI_STATUS_IGNORE);    
    }
    
    MPI_File_close(&output_file);
    
    delete buf;
    delete data;
    delete tmp;

    MPI_Finalize();
    return 0;
}
