#ifndef MODELS_CONNECTIONS_NDN_2016
#define MODELS_CONNECTIONS_NDN_2016

// Синаптический ток спадает по экспоненте
void I_synaptic_exp(

    // --- входные параметры
    // число нейронов
    int _N_neur,
    // массив значений вспомогательных модулирующих переменных 
    // на предыдущем временном слое  
    float *_V_arr,
    // шаг по времени
    float *_dt, 
    // характерное время спада синаптического тока
    float *_tau,
    // пределельное значение потенциала
    float *_V_lim,
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
    float *_y_curr_arr

);

void make_rand_connections(

    // --- входные параметры
    
    // число нейронов
    int _N_neur,
    // число возбуждающих нейронов
    int _N_exc_neur,

    // число связей
    int _N_con,
    // минимально возможный вес свзяи
    float *_min_weight_con,
    // максимально возможный вес свзяи
    float *_max_weight_con,
    // уникальный идентификатор для инициализации srand
    int _uniq,

    // --- выходные параметры
    // массив с номерами предсинаптических нейронов
    int *_pre_con_arr,
    // массив с номерами постсинаптических нейронов
    int *_post_con_arr,
    // массив со значениями весов для каждой связи
    float *_weights_con_arr

);

#endif