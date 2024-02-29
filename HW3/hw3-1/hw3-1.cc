#include <pthread.h>
#include <stdio.h>
#include <math.h>
#include <sched.h>
#include <algorithm>

int **dist; 
int v_num,cpu_num;

pthread_barrier_t barrier;
const int INF = (1 << 30) - 1;

struct t_arg{
	int id;
	int num_threads;
	int p;
};

void* floydwarshall(void* targ) {
	t_arg* arg = (t_arg*)targ;
    int id = arg->id;
    int p = arg->p;
    int start,end,x,y;
    //printf("id=%d,p=%d\n", id,p);

    //printf("v_numcpu_num=%d\n", v_num%cpu_num);
    //handle剛剛算p的餘數
    if(id < v_num%cpu_num) {
        x = id;
        y = 1;
        //printf("x=%d,y=%d\n", x,y);
    }else{
        x = v_num%cpu_num; 
        y = 0;
    } 
    //printf("x=%d,y=%d\n", x,y);
    start = id * p + x;
    end = start + p + y;
    for (int k = 0; k < v_num; ++k)
	{
		for (int i = start; i < end; ++i)
		{
			for (int j = 0; j < v_num; ++j)
			{
				if (dist[i][j] > dist[i][k] + dist[k][j] && dist[i][k] != INF)
					dist[i][j] = dist[i][k] + dist[k][j];
			}
		}
        pthread_barrier_wait(&barrier);
	}
    
    return NULL;
}

int main(int argc, char **argv)
{
	
	if (argc != 3)
		return 0;
    cpu_set_t cpu_set;
    sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
    cpu_num = CPU_COUNT(&cpu_set);
    
    pthread_barrier_init(&barrier, NULL, cpu_num);

	int e_num;
	
    FILE* infile = fopen(argv[1], "rb");
    fread(&v_num, sizeof(int), 1, infile);
    fread(&e_num, sizeof(int), 1, infile);
    //printf("cpu=%d,vnum=%d\n", cpu_num,v_num);
	
    dist = (int **)malloc(v_num * sizeof(int *));
    for (int i = 0; i < v_num; ++i)
    {
        dist[i] = (int *)malloc(v_num * sizeof(int));
    }

    //int p = ceil(v_num / cpu_num);
    int p = v_num/cpu_num; //每個thread至少要負責的量
    //printf("p=%d",p);

    //#pragma omp parallel for schedule(static, p) num_threads(cpu_num)
	for (int i = 0; i < v_num; ++i)
    {
        for (int j = 0; j < v_num; ++j)
        {
            if(i!=j){
                dist[i][j] = INF;
            }
            // }else{
            //     dist[i][j] = INF;
            // }
        //dist[i][i] = 0;
        }  
    } 
    int src, dst, w;
	while (fread(&src, sizeof(int), 1, infile) &&
        fread(&dst, sizeof(int), 1, infile) &&
        fread(&w, sizeof(int), 1, infile))
    {
        dist[src][dst] = w;
    }

    fclose(infile);

    pthread_t threads[cpu_num];
    t_arg args[cpu_num];
    for (int t = 0; t < cpu_num; t++) {
		args[t].id = t;
		args[t].num_threads = cpu_num;
		args[t].p = p;
        //printf("num=%d\n",  args[t].num_threads);
		
        int rc = pthread_create(&threads[t], NULL, floydwarshall , (void*)&args[t]);
        if (rc) {
            printf("ERROR; return code from pthread_create() is %d\n", rc);
            exit(-1);
        }
		
    }
	for (int t = 0; t < cpu_num; t++) {
		pthread_join(threads[t], NULL);
	}
    
    //output
    FILE *outfile = fopen(argv[2], "wb");
    for (int i = 0; i < v_num; ++i) {
        for (int j = 0; j < v_num; ++j) {
            fwrite(&dist[i][j], sizeof(int), 1, outfile);
        }
    }

    fclose(outfile);

    pthread_exit(NULL);
    return 0;
}
