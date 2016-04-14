   // массив для хранения потенцилов всех нейронов V
    // V == V_curr при численном решении СДУ
    float * V = (float*)malloc(N_neur*sizeof(float));
    if( !V ){
        puts("\033[31;1mОшибка выделения памяти (V) \033[30;0m ");
        MPI_Abort(MPI_COMM_WORLD,1);
    }

    // V и u на  узле
    float * V_node = (float*)malloc(N_neur_node*sizeof(float));
    if( !V_node ){
        puts("\033[31;1mОшибка выделения памяти (V_node) \033[30;0m ");
        MPI_Abort(MPI_COMM_WORLD,1);
    }
    float * u_node = (float*)malloc(N_neur_node*sizeof(float));
    if( !u_node ){
        puts("\033[31;1mОшибка выделения памяти (u_node) \033[30;0m ");
        MPI_Abort(MPI_COMM_WORLD,1);
    }


    // часть массива значений синаптических токов на узле 
    float * I_syn_node = (float*)malloc(N_neur_node*sizeof(float));
    if( !I_syn_node ){
        puts("\033[31;1mОшибка выделения памяти (I_syn_node) \033[30;0m ");
        MPI_Abort(MPI_COMM_WORLD,1);
    }

    // массив частичных токов для всех нейронах, распределенных по узлам 
    float * part_I_syn_node = (float*)malloc(N_neur*sizeof(float));
    if( !part_I_syn_node ){
        puts("\033[31;1mОшибка выделения памяти (part_I_syn_node) \033[30;0m ");
        MPI_Abort(MPI_COMM_WORLD,1);
    }

    // массив для хранения внешних токов для каждого нейрона
    float * I_ext = (float*)malloc(N_neur*sizeof(float));
    if( !I_ext ){
        puts("\033[31;1mОшибка выделения памяти (I_ext) \033[30;0m ");
        MPI_Abort(MPI_COMM_WORLD,1);
    }

    float * I_ext_node = (float*)malloc(N_neur_node*sizeof(float));
    if( !I_ext_node ){
        puts("\033[31;1mОшибка выделения памяти (I_ext_node) \033[30;0m ");
        MPI_Abort(MPI_COMM_WORLD,1);
    }

    // --- для реализации метода Рунге-Кутты
    // массивы со значениями с предыдущего временного шага
    float * V_prev = (float*)malloc(N_neur*sizeof(float));
    if( !V_prev ){
        puts("\033[31;1mОшибка выделения памяти (V_prev) \033[30;0m ");
        MPI_Abort(MPI_COMM_WORLD,1);
    }

    float * u_prev = (float*)malloc(N_neur*sizeof(float));
    if( !u_prev ){
        puts("\033[31;1mОшибка выделения памяти (u_prev) \033[30;0m ");
        MPI_Abort(MPI_COMM_WORLD,1);
    }


    // V_prev и u_prev на узле
    float * V_prev_node = (float*)malloc(N_neur_node*sizeof(float));
    if( !V_prev_node ){
        puts("\033[31;1mОшибка выделения памяти (V_prev_node) \033[30;0m ");
        MPI_Abort(MPI_COMM_WORLD,1);
    }
    float * u_prev_node = (float*)malloc(N_neur_node*sizeof(float));
    if( !u_prev_node ){
        puts("\033[31;1mОшибка выделения памяти (u_prev_node) \033[30;0m ");
        MPI_Abort(MPI_COMM_WORLD,1);
    }


    // вспомогательный массив
    float *k_V_node = (float*)malloc(N_neur_node*sizeof(float));
    if( !k_V_node ){
        puts("\033[31;1mОшибка выделения памяти (k_V_node) \033[30;0m ");
        MPI_Abort(MPI_COMM_WORLD,1);
    }

    float *k_u_node = (float*)malloc(N_neur_node*sizeof(float));
    if( !k_u_node ){
        puts("\033[31;1mОшибка выделения памяти (k_u_node) \033[30;0m ");
        MPI_Abort(MPI_COMM_WORLD,1);
    }

    // вспомогательные массивы для последовательности вычислений
    // в методе Рунге-Кутты
    float *V_tmp_node = (float*)malloc(N_neur_node*sizeof(float));
    if( !V_tmp_node ){
        puts("\033[31;1mОшибка выделения памяти (V_tmp_node) \033[30;0m ");
        MPI_Abort(MPI_COMM_WORLD,1);
    }

    float *u_tmp_node = (float*)malloc(N_neur_node*sizeof(float));
    if( !u_tmp_node ){
        puts("\033[31;1mОшибка выделения памяти (u_tmp_node) \033[30;0m ");
        MPI_Abort(MPI_COMM_WORLD,1);
    }

    // переменные для формирования связей в нейронной сети

    // массив хранит индексы предсинаптических нейронов
    int *pre_con_node = (int*)malloc(N_con_node*sizeof(int));
    if( !pre_con_node){
        puts("\033[31;1mОшибка выделения памяти (pre_con_node) \033[30;0m");
        MPI_Abort(MPI_COMM_WORLD,1);
    }

    // массив хранит индексы постсинаптических нейронов
    int *post_con_node = (int*)malloc(N_con_node*sizeof(int));
    if( !post_con_node ){
        puts("\033[31;1mОшибка выделения памяти (post_con_node) \033[30;0m ");
        MPI_Abort(MPI_COMM_WORLD,1);
    }

    // массив хранит значения весов связей
    float *weights_con_node = (float*)malloc(N_con_node*sizeof(float));
    if( !weights_con_node ){
        puts("\033[31;1mОшибка выделения памяти (weights_con_node) \033[30;0m ");
        MPI_Abort(MPI_COMM_WORLD,1);
    }

    float *y_con_prev_node = (float*)malloc(N_con_node*sizeof(float));
    if( !y_con_prev_node ){
        puts("\033[31;1mОшибка выделения памяти (y_con_prev_node) \033[30;0m ");
        MPI_Abort(MPI_COMM_WORLD,1);
    }

    float *y_con_curr_node = (float*)malloc(N_con_node*sizeof(float));
    if( !y_con_curr_node ){
        puts("\033[31;1mОшибка выделения памяти (y_con_curr_node) \033[30;0m ");
        MPI_Abort(MPI_COMM_WORLD,1);
    }

    // --- *_spike массивы для хранения спайков
    
    // neur_spike и t_spike здесь определять нельзя из-за отсуствия значения для  N_spike

    int *neur_spike_znode = NULL; // = (int*)malloc(N_spike*sizeof(int));

    float *t_spike_znode = NULL; // = (float*)malloc(N_spike*sizeof(float));

    // на узле
    int *neur_spike_node = (int*)malloc(N_spike_node*sizeof(int));
    if( !neur_spike_node ){
        puts("\033[31;1mОшибка выделения памяти (neur_spike_node) \033[30;0m ");
        MPI_Abort(MPI_COMM_WORLD,1);
    }

    float *t_spike_node = (float*)malloc(N_spike_node*sizeof(float));
    if( !t_spike_node ){
        puts("\033[31;1mОшибка выделения памяти (t_spike_node) \033[30;0m ");
        MPI_Abort(MPI_COMM_WORLD,1);
    }
