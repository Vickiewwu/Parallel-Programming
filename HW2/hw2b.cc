#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define PNG_NO_SETJMP
#include <sched.h>
#include <assert.h>
#include <png.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <omp.h>
#include <math.h>


void write_png(const char* filename, int iters, int width, int height, const int* buffer, const int p, const int size) {
    FILE* fp = fopen(filename, "wb");
    assert(fp);
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    assert(png_ptr);
    png_infop info_ptr = png_create_info_struct(png_ptr);
    assert(info_ptr);
    png_init_io(png_ptr, fp);
    png_set_IHDR(png_ptr, info_ptr, width, height, 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_set_filter(png_ptr, 0, PNG_NO_FILTERS);
    png_write_info(png_ptr, info_ptr);
    png_set_compression_level(png_ptr, 1);
    size_t row_size = 3 * width * sizeof(png_byte);
    png_bytep row = (png_bytep)malloc(row_size);
    int y=0;
    for (int i = 0; i < height; ++i) {
        memset(row, 0, row_size);
        for (int x = 0; x < width; ++x) {
            int p = buffer[ y * width + x]; 
            png_bytep color = row + x * 3;
            if (p != iters) {
                if (p & 16) {
                    color[0] = 240;
                    color[1] = color[2] = p % 16 * 16;
                } else {
                    color[0] = p % 16 * 16;
                }
            }
        }
        png_write_row(png_ptr, row);
        y += p;
        if (y >= p * size)
			y = y % p + 1;
    }
    free(row);
    png_write_end(png_ptr, NULL);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
}

int main(int argc, char** argv) {

    int rank, size;
    /* detect how many CPUs are available */
    cpu_set_t cpu_set;
    sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
    //printf("%d cpus available\n", CPU_COUNT(&cpu_set));
    int num_threads = CPU_COUNT(&cpu_set); 
    MPI_File input_file, output_file;
    MPI_Offset offset;

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* argument parsing */
    assert(argc == 9);
    const char* filename = argv[1];
    int iters = strtol(argv[2], 0, 10);
    double left = strtod(argv[3], 0);
    double right = strtod(argv[4], 0);
    double lower = strtod(argv[5], 0);
    double upper = strtod(argv[6], 0);
    int width = strtol(argv[7], 0, 10);
    int height = strtol(argv[8], 0, 10);
    int p;
    int flag=0;
    int nproc = size;

    if(height>=size){
        p = ceil((double)height/size);  //每個process要處理的row數量
    }
    else{  //row數比process數目少 
        p = 1;
        flag = 1; //special case
        nproc = height;  //表示在這個case中要使用的process數量
    }
    

    /* allocate memory for image */
    int* image = (int*)malloc(width * p * size * sizeof(int));
    assert(image);

    /* allocate memory for each process */
    int* imagep = (int*)malloc(width * p * sizeof(int));
    

    /* mandelbrot set */
    double yd = (upper - lower) / height;
    double xd = (right - left) / width;
    int rn = 0;  //the number of row the process dealing with
    
    if(rank<=nproc-1){
        for (int j = height-1-rank; j >=0 ; j-=nproc) {
            double y0 = j * yd + lower;
        #pragma omp parallel for num_threads(num_threads) schedule(dynamic, 1)
            for (int i = 0; i < width; ++i) {
                double x0 = i * xd + left;
                int repeats = 0;
                double x = 0;
                double y = 0;
                double length_squared = 0;
                while (repeats < iters && length_squared < 4) {
                    double temp = x * x - y * y + x0;
                    y = 2 * x * y + y0;
                    x = temp;
                    length_squared = x * x + y * y;
                    ++repeats;
                }
                imagep[rn * width + i] = repeats;
            }
            ++rn;
        }
    }
 

    MPI_Gather(imagep, p * width, MPI_INT, image, p * width, MPI_INT, 0, MPI_COMM_WORLD);
    
    /* draw and cleanup */
    if(rank==0){
        write_png(filename, iters, width, height, image, p, nproc);
    }
    
    free(image);
    free(imagep);
    MPI_Finalize();
    return 0;
}
