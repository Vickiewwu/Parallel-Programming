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
#include <pthread.h>
#include <math.h>
#include <emmintrin.h>


int num_threads, iters, width, height, *image;
double left, right, lower, upper;

void *cal(void *tid){
    int id = *((int *)tid);

    /* mandelbrot set */
    //每個thread進來處理一個row
    double dy = (upper - lower) / height;
    double dx = (right - left) / width;
    __m128d v2 = _mm_set_pd1(2);
	__m128d v4 = _mm_set_pd1(4);
    for (int j = id; j < height; j+=num_threads) {
        //printf("id=%d,j=%d\n",id,j);
        double y0 = j * dy + lower;
        __m128d v_y0 = _mm_load1_pd(&y0);
        
        for (int i = 0; i < width-1; i+=2) {
            double x0[2] = {i * dx + left, (i+1)*dx +left};
            __m128d v_x0 = _mm_load_pd(x0);
            int repeats[2] = {0,0};
            __m128d vx = _mm_set_pd(0, 0);
            __m128d vy = _mm_set_pd(0, 0);
            __m128d v_length_squared = _mm_set_pd(0, 0);
            int end[2] = {0,0};
            while (end[0]!=1 || end[1]!=1) {
                if (end[0]!=1)
					{
						if (repeats[0] < iters && _mm_comilt_sd(v_length_squared, v4))
							++repeats[0];
						else
							end[0] = 1;
					}
					if (end[1]!=1)
					{
						if (repeats[1] < iters && _mm_comilt_sd(_mm_shuffle_pd(v_length_squared, v_length_squared, 1), v4))
							++repeats[1];
						else
							end[1] = 1;
					}
                __m128d temp = _mm_add_pd(_mm_sub_pd(_mm_mul_pd(vx,vx),_mm_mul_pd(vy,vy)),v_x0);
                vy = _mm_add_pd(_mm_mul_pd(_mm_mul_pd(v2,vx),vy),v_y0);
                vx = temp;
                v_length_squared =_mm_add_pd(_mm_mul_pd(vx, vx), _mm_mul_pd(vy, vy));
                
                //printf("ls = %lf",length_squared);
            }
            //double index = j * width + i;
            image[j * width + i] = repeats[0];
            image[j * width + i + 1] = repeats[1];
            
            //printf("index=%lf\n",index);
            //printf("repeat=%d\n",repeats);
        }
        if(width%2==1)
        {
            double x0 = (width-1) * dx + left;
            double x = 0;
            double y = 0;
            double length_squared = 0;
            int repeats = 0;
            while (repeats < iters && length_squared < 4)
            {
                double temp = x * x - y * y + x0;
                y = 2 * x * y + y0;
                x = temp;
                length_squared = x * x + y * y;
                ++repeats;
            }
            image[(j+1)*(width)-1] = repeats;
            //printf("pixel=%d\n",(j+1)*(width)-1);
        }
        //printf("j=%d,height=%d\n",j,height);
        
    }
    
    return NULL;
}
void write_png(const char* filename, int iters, int width, int height, const int* buffer) {
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
    for (int y = 0; y < height; ++y) {
        memset(row, 0, row_size);
        for (int x = 0; x < width; ++x) {
            int p = buffer[(height - 1 - y) * width + x];
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
    }
    free(row);
    png_write_end(png_ptr, NULL);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
}

int main(int argc, char** argv) {
    /* detect how many CPUs are available */
    cpu_set_t cpu_set;
    sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
    printf("%d cpus available\n", CPU_COUNT(&cpu_set));
    num_threads = CPU_COUNT(&cpu_set); //=ncpus
    
	pthread_t threads[num_threads];
    int id[num_threads];
   
	
   
    /* argument parsing */
    assert(argc == 9);
    const char* filename = argv[1];
    iters = strtol(argv[2], 0, 10);
    left = strtod(argv[3], 0);
    right = strtod(argv[4], 0);
    lower = strtod(argv[5], 0);
    upper = strtod(argv[6], 0);
    width = strtol(argv[7], 0, 10);
    height = strtol(argv[8], 0, 10);
    
    /* allocate memory for image */
    image = (int*)malloc(width * height * sizeof(int));
    assert(image);
    

    for (int t = 0; t < num_threads; t++) {
        id[t] = t;
        pthread_create(&threads[t], NULL, cal, (void*)&id[t]);
    }
    

    for (int t = 0; t < num_threads; t++) {
		pthread_join(threads[t], NULL);
	}

    /* draw and cleanup */
    write_png(filename, iters, width, height, image);

    pthread_exit(NULL);
    free(image);
    return 0;
}
