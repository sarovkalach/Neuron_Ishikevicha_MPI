#include <stdio.h>

__global__ void F_Ishikevich_Kernel(

    // in:
    int _N_neur, 
    float *_V, 
    float *_u, 
    float *_I_syn,
    float *_I_ext,
    float _C,
    float _V_a, 
    float _V_b, 
    float _V_c,
    float _u_a, 
    float _u_b,

    // out:
    float *_res_V, 
    float *_res_u
)
{
    int __j = blockIdx.x * blockDim.x + threadIdx.x;
    if( __j < _N_neur  ){
        _res_V[__j] = 1. / _C * ( _V_a * _V[__j] * _V[__j] + _V_b * _V[__j] + _V_c - _u[__j] + _I_ext[__j] + _I_syn[__j] );
        _res_u[__j] = _u_a * ( _u_b * _V[__j] - _u[__j] );
    }

}

extern "C"
void F_Ishikevich_GPU(

    int _N_neur, 
    float *_V, 
    float *_u,  
    float *_I_syn,
    float *_I_ext,
    float *_C,
    float *_V_a, 
    float *_V_b, 
    float *_V_c,
    float *_u_a, 
    float *_u_b,
    
    float *_res_V, 
    float *_res_u

){

    //int dev = 0;
    //cudaDeviceProp deviceProp;
    //cudaGetDeviceProperties(&deviceProp, dev);
    //cudaSetDevice(dev);

    // вынести во входные параметры
    int __block_size = 1024;
    int __grid_size = ( _N_neur + __block_size -1 ) / __block_size ; 
    int __bytes = _N_neur * sizeof( float );

    float *__V_dev = NULL;
    cudaMalloc( (void**) &__V_dev, __bytes );

    float *__u_dev = NULL;
    cudaMalloc( (void**) &__u_dev, __bytes );

    float *__I_syn_dev = NULL;
    cudaMalloc( (void**) &__I_syn_dev, __bytes );

    float *__I_ext_dev = NULL;
    cudaMalloc( (void**) &__I_ext_dev, __bytes );

    float *__res_V_dev = NULL;
    cudaMalloc( (void**) &__res_V_dev, __bytes );

    float *__res_u_dev = NULL;
    cudaMalloc( (void**) &__res_u_dev, __bytes );

    cudaMemcpy( __V_dev, _V, __bytes, cudaMemcpyHostToDevice );
    cudaMemcpy( __u_dev, _u, __bytes, cudaMemcpyHostToDevice );
    cudaMemcpy( __I_syn_dev, _I_syn, __bytes, cudaMemcpyHostToDevice );
    cudaMemcpy( __I_ext_dev, _I_ext, __bytes, cudaMemcpyHostToDevice );

    F_Ishikevich_Kernel<<<__grid_size, __block_size>>>( _N_neur, __V_dev, __u_dev, __I_syn_dev, __I_ext_dev, *_C,  *_V_a, *_V_b, *_V_c, *_u_a, *_u_b, __res_V_dev, __res_u_dev  );

    //cudaDeviceSynchronize();
    //cudaGetLastError();

    cudaMemcpy( _res_V, __res_V_dev, __bytes, cudaMemcpyDeviceToHost );
    cudaMemcpy( _res_u, __res_u_dev, __bytes, cudaMemcpyDeviceToHost );


//printf( "N_neur = %d __grid_size = %d __block_size =%d _V[0] = %f _res_V[0]=%f _u[0]=%f _res_u[0]=%f \n" , _N_neur , __grid_size, __block_size, _V[0], _res_V[0], _u[0], _res_u[0] );

    cudaFree(__V_dev);
    cudaFree(__u_dev);
    cudaFree(__res_V_dev);
    cudaFree(__res_u_dev);

    //cudaDeviceReset();
}

