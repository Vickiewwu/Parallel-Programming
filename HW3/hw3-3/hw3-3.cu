#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

const int INF = ((1 << 30) - 1);
const int B = 64;
int v_num, e_num, N;
int* h_dist;

void input(const char* infile) {  
	FILE* file = fopen(infile, "rb");
	fread(&v_num, sizeof(int), 1, file);
	fread(&e_num, sizeof(int), 1, file);
	printf("v=%d\n", v_num);
	N = v_num;
	if (N % B != 0) {
		N = N + (B - N % B);
	}
	
	h_dist = (int*)malloc(N * N * sizeof(int));
	
	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < N; ++j) {
			if (i==j && i < v_num)
				h_dist[i*N+j] = 0;
			else
				h_dist[i*N+j] = INF;
		}
	}

	int pair[3];
	for (int i = 0; i < e_num; ++i)
	{
		fread(pair, sizeof(int), 3, file);
		h_dist[pair[0]*N + pair[1]] = pair[2];
	}

	fclose(file);
}

void output(const char* outfile) {   
	FILE* file = fopen(outfile, "w");

	for (int i = 0; i < v_num; ++i) {		
		fwrite(&h_dist[i*N], sizeof(int), v_num, file);
	}

	fclose(file);
}

__global__ void phase1(int* d_dist, int r, int N) {  //解掉bank conflict 
	
	__shared__ int s[B][B];

	int x = threadIdx.x;
	int y = threadIdx.y;

	int i = r * B + y;
	int j = r * B + x;

	s[x][y] = d_dist[i * N + j];
	s[x][y + 32] = d_dist[(i + 32) * N + j];
	s[x + 32][y] = d_dist[i * N + (j + 32)];
	s[x + 32][y + 32] = d_dist[(i + 32) * N + (j + 32)];

#pragma unroll 32
	for (int k = 0; k < B; ++k) {
		__syncthreads();

		s[x][y] = min(s[x][y], s[k][y] + s[x][k]);
		s[x][y + 32] = min(s[x][y + 32], s[k][y + 32] + s[x][k]);
		s[x + 32][y] = min(s[x + 32][y], s[k][y] + s[x + 32][k]);
		s[x + 32][y + 32] = min(s[x + 32][y + 32], s[k][y + 32] + s[x + 32][k]);
	}

	d_dist[i * N + j] = s[x][y];
	d_dist[(i + 32) * N + j] = s[x][y + 32];
	d_dist[i * N + (j + 32)] = s[x + 32][y];
	d_dist[(i + 32) * N + (j + 32)] = s[x + 32][y + 32];
}
__global__ void phase2_1(int* d_dist, int r, int N) {
	
	__shared__ int s1[B][B];
	__shared__ int s2[B][B];

	int x = threadIdx.x;
	int y = threadIdx.y;

	// pivot
	int i = r * B + y;
	int j = r * B + x;

	s1[y][x] = d_dist[i * N + j];
	s1[y + 32][x] = d_dist[(i + 32) * N + j];
	s1[y][x + 32] = d_dist[i * N + (j + 32)];
	s1[y + 32][x + 32] = d_dist[(i + 32) * N + (j + 32)];

	// row
	//i = r * B + y;    
	j = blockIdx.x * B + x;

	s2[y][x] = d_dist[i * N + j];
	s2[y + 32][x] = d_dist[(i + 32) * N + j];
	s2[y][x + 32] = d_dist[i * N + (j + 32)];
	s2[y + 32][x + 32] = d_dist[(i + 32) * N + (j + 32)];

	__syncthreads();

#pragma unroll 32
	for (int k = 0; k < B; ++k) {		
		s2[y][x] = min(s2[y][x], s1[y][k] + s2[k][x]);
		s2[y + 32][x] = min(s2[y + 32][x], s1[y + 32][k] + s2[k][x]);
		s2[y][x + 32] = min(s2[y][x + 32], s1[y][k] + s2[k][x + 32]);
		s2[y + 32][x + 32] = min(s2[y + 32][x + 32], s1[y + 32][k] + s2[k][x + 32]);

	}
	d_dist[i * N + j] = s2[y][x];
	d_dist[(i + 32) * N + j] = s2[y + 32][x];
	d_dist[i * N + (j + 32)] = s2[y][x + 32];
	d_dist[(i + 32) * N + (j + 32)] = s2[y + 32][x + 32];
}

__global__ void phase2_2(int* d_dist, int r, int N) {
	
	__shared__ int s1[B][B];
	__shared__ int s2[B][B];

	int x = threadIdx.x;
	int y = threadIdx.y;

	// pivot
	int i = r * B + y;
	int j = r * B + x;

	s1[y][x] = d_dist[i * N + j];
	s1[y + 32][x] = d_dist[(i + 32) * N + j];
	s1[y][x + 32] = d_dist[i * N + (j + 32)];
	s1[y + 32][x + 32] = d_dist[(i + 32) * N + (j + 32)];

	// col
	i = blockIdx.x * B + y;
	j = r * B + x;

	s2[y][x] = d_dist[i * N + j];
	s2[y + 32][x] = d_dist[(i + 32) * N + j];
	s2[y][x + 32] = d_dist[i * N + (j + 32)];
	s2[y + 32][x + 32] = d_dist[(i + 32) * N + (j + 32)];

	__syncthreads();

#pragma unroll 32
	for (int k = 0; k < B; ++k) {
		s2[y][x] = min(s2[y][x], s2[y][k] + s1[k][x]);
		s2[y + 32][x] = min(s2[y + 32][x], s2[y + 32][k] + s1[k][x]);
		s2[y][x + 32] = min(s2[y][x + 32], s2[y][k] + s1[k][x + 32]);
		s2[y + 32][x + 32] = min(s2[y + 32][x + 32], s2[y + 32][k] + s1[k][x + 32]);

	}

	d_dist[i * N + j] = s2[y][x];
	d_dist[(i + 32) * N + j] = s2[y + 32][x];
	d_dist[i * N + (j + 32)] = s2[y][x + 32];
	d_dist[(i + 32) * N + (j + 32)] = s2[y + 32][x + 32];
}

__global__ void phase3(int* d_dist, int r, int N, int offset) {
	
	__shared__ int s1[B][B];   //self
	__shared__ int s2[B][B];  //row
	__shared__ int s3[B][B];   //col

	int x = threadIdx.x;
	int y = threadIdx.y;

	int i; 
	int j; 

	//row
	i = r * B + y;
	j = blockIdx.x * B + x;

	s2[y][x] = d_dist[i * N + j];
	s2[y + 32][x] = d_dist[(i + 32) * N + j];
	s2[y][x + 32] = d_dist[i * N + (j + 32)];
	s2[y + 32][x + 32] = d_dist[(i + 32) * N + (j + 32)];

	i = (blockIdx.y + offset) * B + y;
	j = r * B + x;

	s3[y][x] = d_dist[i * N + j];
	s3[y + 32][x] = d_dist[(i + 32) * N + j];
	s3[y][x + 32] = d_dist[i * N + (j + 32)];
	s3[y + 32][x + 32] = d_dist[(i + 32) * N + (j + 32)];

	__syncthreads();

	i = (blockIdx.y + offset) * B + y;
	j = blockIdx.x * B + x;

	s1[y][x] = d_dist[i * N + j];
	s1[y + 32][x] = d_dist[(i + 32) * N + j];
	s1[y][x + 32] = d_dist[i * N + (j + 32)];
	s1[y + 32][x + 32] = d_dist[(i + 32) * N + (j + 32)];

#pragma unroll 32
	for (int k = 0; k < B; ++k) {
		s1[y][x] = min(s1[y][x], s3[y][k] + s2[k][x]);
		s1[y + 32][x] = min(s1[y + 32][x], s3[y + 32][k] + s2[k][x]);
		s1[y][x + 32] = min(s1[y][x + 32], s3[y][k] + s2[k][x + 32]);
		s1[y + 32][x + 32] = min(s1[y + 32][x + 32], s3[y + 32][k] + s2[k][x + 32]);
	}

	d_dist[i * N + j] = s1[y][x];
	d_dist[(i + 32) * N + j] = s1[y + 32][x];
	d_dist[i * N + (j + 32)] = s1[y][x + 32];
	d_dist[(i + 32) * N + (j + 32)] = s1[y + 32][x + 32];
}

int main(int argc, char** argv) {

	input(argv[1]);

	int* d_dist[2];
	
	int gridsize = N / B;
	int p = gridsize / 2;	
	

#pragma omp parallel num_threads(2)
	{
		int id = omp_get_thread_num();

		cudaSetDevice(id);
		cudaStream_t stream[2];
    	cudaStreamCreate(&stream[0]);
    	cudaStreamCreate(&stream[1]);

		dim3 grid2(gridsize, 1);
		dim3 block(32, 32);
		dim3 grid3;
		int ydim;

		if(gridsize %2 !=0){
			grid3 = dim3(gridsize, p+id);
			ydim = p+id;
		}else{
			grid3 = dim3(gridsize, p);
			ydim = p;
		}
		
		int offset = id * p;

		cudaMalloc((void**)&d_dist[id], N * N * sizeof(int));
#pragma omp barrier
		cudaMemcpy(&d_dist[id][offset *  N * B], &h_dist[offset *  N * B], ydim *  N * B * sizeof(int), cudaMemcpyHostToDevice);

		for (int r = 0; r < gridsize; ++r) {
			
			if (offset <= r && r < offset + ydim)
			{	
				int index = r * N * B;
				if (id == 0)
					cudaMemcpy(&(d_dist[1][index]), &(d_dist[0][index]),
							   sizeof(int) * B * N, cudaMemcpyDeviceToDevice);
				else
					cudaMemcpy(&(d_dist[0][index]), &(d_dist[1][index]),
							   sizeof(int) * B * N, cudaMemcpyDeviceToDevice);
			}
#pragma omp barrier
			phase1<<<1, block>>>(d_dist[id], r, N);
			phase2_1<<<grid2, block,0,stream[0]>>>(d_dist[id], r, N);
			phase2_2<<<grid2, block,0,stream[1]>>>(d_dist[id], r, N);
			phase3<<<grid3, block>>>(d_dist[id], r, N, offset);
		}

		cudaMemcpy(&h_dist[offset * N * B], &d_dist[id][offset * N * B], ydim * N * B * sizeof(int), cudaMemcpyDeviceToHost);

	}
	
	output(argv[2]);

	return 0;
}
