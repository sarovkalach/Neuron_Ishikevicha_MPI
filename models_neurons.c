#include <stdlib.h>
#include <time.h>
#include <omp.h>

#include "models_neurons.h"

void F_Ishikevich(

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

    // счетчик для перебора нейронов
    int __j;

    // вычисление правых частей СДУ
    #pragma omp parallel for private(__j) default(shared)
    for( __j = 0; __j < _N_neur; ++__j ){

        _res_V[__j] = 1. / *_C * ( *_V_a * _V[__j] * _V[__j] + *_V_b * _V[__j] + *_V_c - _u[__j] + _I_ext[__j] + _I_syn[__j] );

        _res_u[__j] = *_u_a * ( *_u_b * _V[__j] - _u[__j] );

    }

}

void make_rand_external_current(

    int _N_neur,
    float *_I_ext_max,
    int _prec,
    int _uniq,

    float *_I_ext

){

    int __i;


    #pragma omp parallel default(shared)
    {
        srand( time(NULL) + _uniq + omp_get_thread_num() );
        //srand( time(NULL) + _uniq  );
        #pragma omp  for private(__i) 
        for ( __i = 0; __i < _N_neur; ++__i ){
            _I_ext[__i] = (rand() % (int) (_prec*(*_I_ext_max)))/(double)_prec;

        }
    }

}
