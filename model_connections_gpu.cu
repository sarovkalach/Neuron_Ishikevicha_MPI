#include <stdio.h>
#include "common.h"
#include <math.h>

int dev = 0;
cudaError_t t =	cudaSetDevice(dev);
cudaDeviceProp deviceProp;
cudaError_t p =  cudaGetDeviceProperties(&deviceProp, dev);
float mem = deviceProp.totalGlobalMem;

__constant__ float   dt = 0.1;
__constant__ float   V_lim = 30;
__constant__ float   tau = 4.0;


__device__ void I_synaptic_exp_kernel2(float* ptr, float value, int _shift)
{
	*ptr = atomicAdd(&(ptr[_shift]), value);
}

__global__  void I_synaptic_exp_kernel(

	float* V,
	int _N_con_chunk,
	float* y_prev,
	int* pre_con,
	int* post_con,
	float* weights_con,

	float* I_syn,
	float* y_curr

)

{
	unsigned int ix = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int iy = blockIdx.y*blockDim.y + threadIdx.y;
	unsigned int idx = (iy * (_N_con_chunk) + ix);


	if (( ix < _N_con_chunk) && (iy < _N_con_chunk))
	{
		y_curr[idx] = y_prev[idx] * __expf(-dt/tau);
		if (V[pre_con[idx]] > V_lim) y_curr[idx] = 1.0;

		float tmp = y_curr[idx] * weights_con[idx];
		int shift = post_con[idx];

		I_synaptic_exp_kernel2(I_syn, tmp, shift);

	}

}




extern "C"
void I_synaptic_exp_gpu(
    int _N_neur,
    float *_V_arr,
    float _dt,
    float _tau,
    float _V_lim,
    int _N_con,
    float *_y_prev_arr,
    int *_pre_con_arr,
    int *_post_con_arr,
    float *_weights_arr,
    // out:
    float *_I_syn_arr,
    float *_y_curr_arr,
    int size_block_x,
    int size_block_y

)
{

	dim3 block(size_block_x, size_block_y);
	int _N_con_x = sqrt(_N_con);
	int _N_con_y = sqrt(_N_con);
	dim3 grid((_N_con_x +block.x -1)/block.x, (_N_con_y +block.y -1)/block.y);

	memset(_I_syn_arr, 0, _N_neur*sizeof(float));

	float* I_syn_dev;
    CHECK(cudaMalloc((void **)&I_syn_dev, _N_neur*sizeof(float)));
    CHECK(cudaMemcpy(I_syn_dev, _I_syn_arr, _N_neur*sizeof(float), cudaMemcpyHostToDevice));

    float* V_dev;
    CHECK(cudaMalloc((void **)&V_dev, _N_neur*sizeof(float)));
    CHECK(cudaMemcpy(V_dev, _V_arr, _N_neur*sizeof(float), cudaMemcpyHostToDevice));

    float *y_curr_dev;
    CHECK(cudaMalloc((void **)&y_curr_dev, _N_con*sizeof(float)));
    CHECK(cudaMemcpy(y_curr_dev, _y_curr_arr, _N_con*sizeof(float), cudaMemcpyHostToDevice));

    float *y_prev_dev;
    CHECK(cudaMalloc((void **)&y_prev_dev, _N_con*sizeof(float)));
    CHECK(cudaMemcpy(y_prev_dev, _y_prev_arr, _N_con*sizeof(float), cudaMemcpyHostToDevice));

    int *pre_con_dev;
    CHECK(cudaMalloc((void **)&pre_con_dev, _N_con*sizeof(int)));
    CHECK(cudaMemcpy(pre_con_dev, _pre_con_arr, _N_con*sizeof(int), cudaMemcpyHostToDevice));

    int *post_con_dev;
    CHECK(cudaMalloc((void **)&post_con_dev, _N_con*sizeof(int)));
    CHECK(cudaMemcpy(post_con_dev, _post_con_arr, _N_con*sizeof(int), cudaMemcpyHostToDevice));

    float *weights_con_dev;
    CHECK(cudaMalloc((void **)&weights_con_dev, _N_con*sizeof(float)));
    CHECK(cudaMemcpy(weights_con_dev, _weights_arr, _N_con*sizeof(float), cudaMemcpyHostToDevice));

    I_synaptic_exp_kernel<<<grid, block,_N_neur*sizeof(float)>>>(V_dev,_N_con_x,y_prev_dev,pre_con_dev,post_con_dev,weights_con_dev,I_syn_dev,y_curr_dev);

    CHECK(cudaMemcpy(_I_syn_arr, I_syn_dev, _N_neur*sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(_V_arr, V_dev, _N_neur*sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(_y_curr_arr, y_curr_dev, _N_con*sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(_y_prev_arr, y_prev_dev, _N_con*sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(_pre_con_arr, pre_con_dev, _N_con*sizeof(int), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(_post_con_arr, post_con_dev, _N_con*sizeof(int), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(_weights_arr, weights_con_dev, _N_con*sizeof(float), cudaMemcpyDeviceToHost));

    cudaFree(I_syn_dev);
    cudaFree(V_dev);
    cudaFree(y_curr_dev);
    cudaFree(y_prev_dev);
    cudaFree(pre_con_dev);
    cudaFree(post_con_dev);
    cudaFree(weights_con_dev);

}
