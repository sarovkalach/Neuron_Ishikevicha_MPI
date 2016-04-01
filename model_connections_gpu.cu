#include <stdio.h>
#include "common.h"

__device__ void I_synaptic_exp_Kernel2(

	int   Nneur,
	float *V,
	float *dt,
	float *tau,
	float *V_lim,
	int    Ncon,
	float *y_prev,
	int   *pre_con,
	int   *post_con,
	float *weights_con,

	float *I_syn,
	float *y_curr,
	float *ptr_I_syn,
	float temp
)

{

	*ptr_I_syn = atomicAdd(ptr_I_syn, temp);
}

__global__ void I_synaptic_exp_Kernel(

	int   Nneur,
	float *V,
    float *dt,
    float *tau,
    float *V_lim,
    int    Ncon,
    float *y_prev,
    int   *pre_con,
    int   *post_con,
    float *weights_con,

    float *I_syn,
    float *y_curr
)

{
	float tmp;
	unsigned int ix = blockIdx.x*blockDim.x + threadIdx.x;
	if (ix < Ncon)
	{
		y_curr[ix] = y_prev[ix] * exp( -(*dt) / (*tau));
		if (V[pre_con[ix]] > (*V_lim)) { y_curr[ix] = 1.0; }
		float tmp = y_curr[ix] * weights_con[ix];
		I_synaptic_exp_Kernel2(Nneur,V,dt,tau,V_lim,Ncon,y_prev,pre_con,post_con,weights_con,I_syn,y_curr, &(I_syn[ix]), tmp);
	}

}




extern "C"
void I_synaptic_exp_gpu(
    int   Nneur,
	float *V,
    float *dt,
    float *tau,
    float *V_lim,
    int    Ncon,
    float *y_prev,
    int   *pre_con,
    int   *post_con,
    float *weights_con,

    float *I_syn,
    float *y_curr

)


{
	//инициализируем девайс
	int dev = 0;
	cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    cudaSetDevice(dev);

    int bytes = Ncon*sizeof(float);
    int dimx = 32;
    dim3 blockSize(dimx);
    dim3 gridSize((Nneur + blockSize.x - 1) / blockSize.x);

    float *I_syn_dev;
    CHECK(cudaMalloc((void **)&I_syn_dev, Nneur*sizeof(float)));
    cudaMemcpy(I_syn_dev, I_syn, Nneur*sizeof(float), cudaMemcpyHostToDevice);

    float *V_dev;
    CHECK(cudaMalloc((void **)&V_dev, Nneur*sizeof(float)));
    cudaMemcpy(V_dev, V, Nneur*sizeof(float), cudaMemcpyHostToDevice);

    float *y_curr_dev;
    CHECK(cudaMalloc((void **)&y_curr_dev, bytes));
    cudaMemcpy(y_curr_dev, y_curr, bytes, cudaMemcpyHostToDevice);

    float *y_prev_dev;
    CHECK(cudaMalloc((void **)&y_prev_dev, bytes));
    cudaMemcpy(y_prev_dev, y_prev, bytes, cudaMemcpyHostToDevice);

    int *pre_con_dev;
    CHECK(cudaMalloc((void **)&pre_con_dev, bytes));
    cudaMemcpy(pre_con_dev, pre_con, bytes, cudaMemcpyHostToDevice);

    int *post_con_dev;
    CHECK(cudaMalloc((void **)&post_con_dev, bytes));
    cudaMemcpy(post_con_dev, post_con_dev, bytes, cudaMemcpyHostToDevice);

    float *weights_con_dev;
    CHECK(cudaMalloc((void **)&weights_con_dev, bytes));
    cudaMemcpy(weights_con_dev, weights_con, bytes, cudaMemcpyHostToDevice);

    memset(I_syn_dev,0,Nneur*sizeof(float));	// инициализируем массив токов I_syn нулями

    I_synaptic_exp_Kernel<<<gridSize, blockSize>>>(Nneur,V_dev,dt,tau,V_lim,Ncon,y_prev_dev,pre_con_dev,post_con_dev,weights_con_dev,I_syn_dev,y_curr_dev);

    cudaMemcpy(I_syn, I_syn_dev, Nneur*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(V, V_dev, Nneur*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(y_curr, y_curr_dev, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(y_prev, y_prev_dev, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(pre_con, pre_con_dev, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(post_con, post_con_dev, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(weights_con, weights_con_dev, bytes, cudaMemcpyHostToDevice);

    cudaFree(I_syn_dev);
    cudaFree(V_dev);
    cudaFree(y_curr_dev);
    cudaFree(y_prev_dev);
    cudaFree(pre_con_dev);
    cudaFree(post_con_dev);
    cudaFree(weights_con_dev);

    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceReset());


	;
}
