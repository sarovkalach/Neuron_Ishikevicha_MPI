#ifndef MODELS_NEURONS_GPU_NDN_2016
#define MODELS_NEURONS_GPU_NDN_2016

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

);

#endif
