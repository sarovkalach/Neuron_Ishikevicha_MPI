/*
 * model_connections_gpu.h
 *
 *  Created on: Mar 31, 2016
 *      Author: kalach
 */

#ifndef MODEL_CONNECTIONS_GPU_H_
#define MODEL_CONNECTIONS_GPU_H_

// Синаптический ток спадает по экспоненте
void I_synaptic_exp_gpu(

    // --- входные параметры
    // число нейронов
    int _N_neur,
    // массив значений вспомогательных модулирующих переменных
    // на предыдущем временном слое
    float *_V_arr,
    // шаг по времени
    float _dt,
    // характерное время спада синаптического тока
    float _tau,
    // пределельное значение потенциала
    float _V_lim,
    // - связи
    // число связей
    int _N_con,
    // массив с потенциалами на всех нейронах
    float *_y_prev_arr,
    // массив с номерами предсинаптических нейронов
    int *_pre_con_arr,
    // массив с номерами постсинаптических нейронов
    int *_post_con_arr,
    // массив со значениями весов для каждой связи
    float *_weights_con_arr,

    // --- выходные параметры
    // массив с синаптическим током на каждом нейроне
    float *_I_syn_arr,
    // массив значений вспомогательных модулирующих переменных
    // на текущем временном слое
    float *_y_curr_arr,
    int size_block_x,
    int size_block_y

);




#endif /* MODEL_CONNECTIONS_GPU_H_ */
