#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <stdio.h>
#include <omp.h>

#include "models_connections.h"


void I_synaptic_exp(
    int _N_neur,
    float *_V_arr,
    float *_dt, 
    float *_tau,
    float *_V_lim,
    int _N_con,
    float *_y_prev_arr,
    int *_pre_con_arr,
    int *_post_con_arr,
    float *_weights_arr,
    // out:
    float *_I_syn_arr,
    float *_y_curr_arr
){

    // счетчик для перебора нейронов
    int __i;

    // обнуляем ток для накопления 
    #pragma omp  parallel for private(__i) default(shared)
    for( __i = 0; __i < _N_neur; ++__i ) 
        _I_syn_arr[__i] = 0.;

    // счетчик для перебора связей	
    int __c;

    // перебираем все связи между нейронами 
    #pragma omp  parallel for private(__i) default(shared)
    for( __c = 0; __c < _N_con; ++__c ) {

        // ток спадает по экспонециальному закону
        _y_curr_arr[__c] = _y_prev_arr[__c] * exp( -(*_dt) / (*_tau) );

        // если произошел спайк в предсинаптическом нейроне,
        if( _V_arr[_pre_con_arr[__c]] > (*_V_lim) )  _y_curr_arr[__c] = 1.;

        //накапливам значение силы тока
        #pragma omp atomic
        _I_syn_arr[_post_con_arr[__c]] += _y_curr_arr[__c] * _weights_arr[__c];

        // _y_prev_arr[__c] = _y_curr_arr[__c];

    }

}


void make_rand_connections(
    int _N_neur,
    int _N_exc_neur,
    int _N_con,
    float *_min_weight_con,
    float *_max_weight_con,
    int _uniq,
    int *_pre_con_arr,
    int *_post_con_arr,
    float *_weights_con_arr
) {


    int __i;

    #pragma omp parallel default(shared)
    {
        srand( time(NULL) + _uniq + omp_get_thread_num() );
        //srand( time(NULL) + _uniq );
        #pragma omp  for private(__i) 
        for ( __i = 0; __i < _N_con; ++__i ){

             // нейроны для связи выбираем случайно
            _pre_con_arr[__i] = rand() % _N_neur;
            _post_con_arr[__i] = rand() % _N_neur;

            // веса связей тоже определяем случайно
            _weights_con_arr[__i] = (*_min_weight_con) + rand() % (int)( (*_max_weight_con) - (*_min_weight_con) );

            // тормозящие нейроны просто имеют отрицательный вес
            if ( _pre_con_arr[__i] >= _N_exc_neur )
                _weights_con_arr[__i] *= -1;
        
        }// for __i
    } 
}

