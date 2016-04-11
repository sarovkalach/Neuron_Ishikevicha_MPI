#include <mpi.h>
#include <omp.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
//#include <postgresql/libpq-fe.h>


#include "par_utils.h"
#include "models_neurons.h"
#include "models_connections.h"
#include "ode_numerical.h"

#include "models_neurons_gpu.h"
#include "model_connections_gpu.h"


// 1 - этапы, 2 - значения, 3 - дополнительные проверки
#define DEBUG 1
#define VIDEO 0

#define SIZE_BLOCK_X 32		// размеры блока для видеокарт ВРЕМЕННО!!!!!
#define SIZE_BLOCK_Y 16		//

int main(int argc, char* argv[]){


    // число используемых узлов 
    int nodes_number;

    // номер узла 
    int node_number;

    MPI_Status stat;

    MPI_Init(&argc,&argv);

    // получение числа узлов
    MPI_Comm_size(MPI_COMM_WORLD,&nodes_number);

    if(nodes_number<2) {
        puts("\033[31;1mЧисло узлов должно быть не менее двух\033[30;0m ");
        MPI_Abort(MPI_COMM_WORLD,2);
    }

    // получение номера узла
    MPI_Comm_rank(MPI_COMM_WORLD,&node_number);


    //PGconn *conn;

    //PGresult *res;

    // --- входные переменные 

    // -- нейрон

    // емкость нейрона
    float C;

    // параметры из СДУ модели Ижикевича
    float V_a;
    float V_b;
    float V_c;
    float u_a;
    float u_b;
    float u_d;

    
    // шаг по времен   
    float dt;

    // время  окончания симуляции
    float t_end;

    // -- нейронная сеть

    // всего нейронов
    int N_neur;

    // максимальное значение внешнего тока
    float I_ext_max;

    // начальный потенциал
    float V_start;
    // начальное значение переменной u из СДУ Ижикевича
    float u_start;
    // начальное значение синаптического тока в сети
    float I_syn_start;

    // -- связи между нейронами

    // всего связей
    int N_con;

    // число возбуждающих нейронов
    int N_neur_exc;

    // значения диапазонов для весов связей
    float weight_con_min;
    float weight_con_max;

    // начальное значение синаптического тока в сети
    float y_con_start;

    // характерное время спада тока
    float tau;

    // предельный потенциал
    float V_lim;

    // потенциал после спайка
    float V_after_spike;

    // число нитей
    int N_omp_threads;

    // вспомогательные переменные
    int i, j;

    // число найденных спайков на всех узлах
    int N_spike;

    // первоначальная размерность массива для хранения спайков на узле
    int N_spike_node = 100;

    // число найденных спайков на узле
    int idx_spike_node = 0;

    // флаг использовать/неиспользовать GPU
    int isGPU;


    // --- управляющий узел выполняет вспомогательные работы

    if( !node_number ){

#if DEBUG > 0
        puts("\033[1mЧитаем и рассылаем параметры из БД  \033[30;0m");
#endif

/*          
        int id = atoi(argv[1]);

        conn = PQconnectdb(
            "dbname=atlas host=1.2.3.4 port=5432 user=123 password=123"
        );

        if (PQstatus(conn) != CONNECTION_OK){

            printf("Соединение с базой данных не удалось: %s", PQerrorMessage(conn));
            PQfinish(conn);
            exit(1);
        }

        res = PQexec(conn,"SELECT ...");

        if ((!res) || (PQresultStatus(res) != PGRES_TUPLES_OK)) {

             printf("При обращении к базу данных произошла ошибка: %i\n", PQresultStatus(res)); 
             PQclear(res);
             exit(1);

        }

        //resulf_of_select = PQntuples(res);

        //for(int i = 0 ; i<result_of_select; ++i) { 
            //PQgetvalue(res, i, 0)
        //}

#if DEBUG
     puts("результаты выборки");        
#endif

        PQclear(res);
        PQfinish(conn);
*/

        // определяем переменные,
        // в дальнейшем читаем их из БД
        
        N_neur = 100;
        C = 50;
        V_start = -60.;
        u_start = 0.;
        V_a = 0.5;
        V_b = 52.5;
        V_c = 1350.;
        u_a = 0.02;
        u_b = 0.5; 
        u_d = 100.; 

        N_con =1000;
        N_neur_exc = 96;
        weight_con_min = 50.;
        weight_con_max = 100.;

        I_ext_max = 40.;
        I_syn_start = 0.;
        y_con_start = 0.;
        tau = 4.;
        V_lim = 30.;
        V_after_spike = -50.;

        dt = 0.1;
        t_end = 1000/0.5f;

        N_omp_threads = 2;

        isGPU = 0;

        // проверки для входных данных
        // в дальнейшем ошибка будет записана в БД

        if( N_neur <=0 ) {
            puts("\033[31;1mЧисло нейронов должно быть больше нуля\033[30;0m ");
            MPI_Abort(MPI_COMM_WORLD,3);
        }

        if( N_neur ==1 && N_con != 0 ) {
            puts("\033[31;1mДля одного нейрона требуется указать ноль связей\033[30;0m ");
            MPI_Abort(MPI_COMM_WORLD,3);
        }

        if( N_neur < N_neur_exc ) {
            puts("\033[31;1mЧисло возбуждающих нейронов должно быть не должно превышать общее число нейронов\033[30;0m ");
            MPI_Abort(MPI_COMM_WORLD,3);
        }

        if( N_neur < N_neur_exc) {
            puts("\033[31;1mЧисло возбуждающих нейронов должно быть не должно превышать общее число нейронов\033[30;0m ");
            MPI_Abort(MPI_COMM_WORLD,3);
        }

        if( dt <=0 ) {
            puts("\033[31;1mШаг по времени должен быть положительным\033[30;0m ");
            MPI_Abort(MPI_COMM_WORLD,3);
        }


    }// if_rank==0


    // -----  

   // --- рассылка значений, 
   // прочитанных для входных переменных
   // от управляющего узла всем узлам
   // (по окончанию выделения входных переменных сделать
   // пересылку одной структуры)

    MPI_Bcast(&N_neur,1,MPI_INT,0,MPI_COMM_WORLD);
    MPI_Bcast(&C,1,MPI_FLOAT,0,MPI_COMM_WORLD);
    MPI_Bcast(&V_start,1,MPI_FLOAT,0,MPI_COMM_WORLD);
    MPI_Bcast(&u_start,1,MPI_FLOAT,0,MPI_COMM_WORLD);
    MPI_Bcast(&V_a,1,MPI_FLOAT,0,MPI_COMM_WORLD);
    MPI_Bcast(&V_b,1,MPI_FLOAT,0,MPI_COMM_WORLD);
    MPI_Bcast(&V_c,1,MPI_FLOAT,0,MPI_COMM_WORLD);
    MPI_Bcast(&u_a,1,MPI_FLOAT,0,MPI_COMM_WORLD);
    MPI_Bcast(&u_b,1,MPI_FLOAT,0,MPI_COMM_WORLD);
    MPI_Bcast(&u_d,1,MPI_FLOAT,0,MPI_COMM_WORLD);

    MPI_Bcast(&N_con,1,MPI_INT,0,MPI_COMM_WORLD);
    MPI_Bcast(&N_neur_exc,1,MPI_INT,0,MPI_COMM_WORLD);
    MPI_Bcast(&weight_con_min,1,MPI_FLOAT,0,MPI_COMM_WORLD);
    MPI_Bcast(&weight_con_max,1,MPI_FLOAT,0,MPI_COMM_WORLD);

    MPI_Bcast(&I_ext_max,1,MPI_FLOAT,0,MPI_COMM_WORLD);
    MPI_Bcast(&I_syn_start,1,MPI_FLOAT,0,MPI_COMM_WORLD);
    MPI_Bcast(&y_con_start,1,MPI_FLOAT,0,MPI_COMM_WORLD);
    MPI_Bcast(&tau,1,MPI_FLOAT,0,MPI_COMM_WORLD);
    MPI_Bcast(&V_lim,1,MPI_FLOAT,0,MPI_COMM_WORLD);
    MPI_Bcast(&V_after_spike,1,MPI_FLOAT,0,MPI_COMM_WORLD);

    MPI_Bcast(&dt,1,MPI_FLOAT,0,MPI_COMM_WORLD);
    MPI_Bcast(&t_end,1,MPI_FLOAT,0,MPI_COMM_WORLD);

    MPI_Bcast(&N_omp_threads,1,MPI_INT,0,MPI_COMM_WORLD);

    MPI_Bcast(&isGPU,1,MPI_INT,0,MPI_COMM_WORLD);

    omp_set_num_threads(N_omp_threads);

#if DEBUG > 0
    if( !node_number )
        puts("\033[1mРасчет и рассылка нагрузки для каждого узла  \033[30;0m");
#endif
    // ----- определяем для узлов порции нейронов 

    // число нейронов обрабатываемых на текущем узле
    int N_neur_node;


    int* node_displ_neur = (int*)malloc(nodes_number*sizeof(int));
    if( !node_displ_neur ){
        puts("\033[31;1mОшибка выделения памяти (node_displ_neur) \033[30;0m ");
        MPI_Abort(MPI_COMM_WORLD,1);
    }

    int* node_count_neur = (int*)malloc(nodes_number*sizeof(int));
    if( !node_count_neur ){
        puts("\033[31;1mОшибка выделения памяти (node_count_neur) \033[30;0m ");
        MPI_Abort(MPI_COMM_WORLD,1);
    }



    // расчет порций нейронов на управляющем узле
    if( !node_number){

        calculate_portions(nodes_number,N_neur,node_count_neur,node_displ_neur,0);

#if DEBUG > 1

        puts("\033[36m    Распределение порций нейронов по узлам: \033[33;0m ");

        printf("\033[36m    содержимое переменной chunk_count: \033[30;0m ");
        for( i = 0; i < nodes_number; ++i)
            printf("%i ", node_count_neur[i]);
        puts("");

        printf("\033[36m    содержимое переменной chunk_map: \033[30;0m ");
        for( i = 0; i < nodes_number; ++i)
            printf("%i ", node_displ_neur[i]);
        puts("");

#endif

    }// if ! node_number

    // ------- определяем для узлов порции связей
    
    // число связей, обрабатываемых на текущем узле
    int N_con_node;

   int* node_displ_con = (int*)malloc(nodes_number*sizeof(int));
    if( !node_displ_con ){
        puts("\033[31;1mОшибка выделения памяти (node_displ_con) \033[30;0m ");
        MPI_Abort(MPI_COMM_WORLD,1);
    }

    int* node_count_con = (int*)malloc(nodes_number*sizeof(int));
    if( !node_count_con ){
        puts("\033[31;1mОшибка выделения памяти (node_count_con) \033[30;0m ");
        MPI_Abort(MPI_COMM_WORLD,1);
    }
    
    // расчет порций связей на управляющем узле
    if( !node_number){

        calculate_portions(nodes_number,N_con,node_count_con,node_displ_con,0);

#if DEBUG > 1

        puts("\033[36m    Распределение порций связей по узлам: \033[33;0m ");

        printf("\033[36m    содержимое переменной node_count_con: \033[30;0m ");
        for( i = 0; i < nodes_number; ++i )
            printf("%i ", node_count_con[i]);
        puts("");

        printf("\033[36m    содержимое переменной node_displ_con: \033[30;0m ");
        for( i = 0; i < nodes_number; ++i)
            printf("%i ", node_displ_con[i]);
        puts("");

        
#endif

    }// if ! node_number

    MPI_Bcast( node_count_neur, nodes_number, MPI_INT, 0, MPI_COMM_WORLD );
    MPI_Bcast( node_displ_neur, nodes_number, MPI_INT, 0, MPI_COMM_WORLD );

    // порция связей для текущего узла с идентификатором node_number
    N_neur_node = node_count_neur[node_number];

    MPI_Bcast( node_count_con, nodes_number, MPI_INT, 0, MPI_COMM_WORLD );
    MPI_Bcast( node_displ_con, nodes_number, MPI_INT, 0, MPI_COMM_WORLD );

    // порция связей для текущего узла с идентификатором node_number
    N_con_node = node_count_con[node_number];

#if DEBUG > 1

    int idx_neur_beg_node = node_displ_neur[node_number];
    int idx_neur_end_node = idx_neur_beg_node + N_neur_node;
    printf("    \033[36m узел \033[33;0m %i \033[36m обрабатывает нейроны\033[33;0m от =%i до <%i\n",
                    node_number,idx_neur_beg_node,idx_neur_end_node);

    int idx_con_beg_node = node_displ_con[node_number];
    int idx_con_end_node = idx_con_beg_node + N_con_node;
    printf("    \033[36m узел \033[33;0m %i \033[36m обрабатывает связи\033[33;0m от =%i до <%i\n",
                    node_number,idx_con_beg_node,idx_con_end_node);

    MPI_Barrier(MPI_COMM_WORLD);

#endif


    
    // --- выделим память для всех массивов
    #include "create_add_mass.c" 

    // --- инициализация

    // инициализируем связи

    // для случайного распределения подойдут независимые расчеты:
    // бросаем последовательно один кубик N раз == одновременно N кубиков
    if( node_number ){

        // случайные связи
        make_rand_connections(
           N_neur,
           N_neur_exc,
           N_con_node,
           &weight_con_min,
           &weight_con_max,
           node_number,
           pre_con_node,
           post_con_node,
           weights_con_node 
        );

#if DEBUG > 1
        printf("\033[36m    Связи: \033[30;0m ");
        for ( i = 0; i < N_con_node; ++i ) 
            printf("(%i->%i weight=%4.1lf) ",pre_con_node[i], post_con_node[i], weights_con_node[i]);
        puts("");
#endif

    } // if node_number


    // инициализируем внешний ток
    if( node_number ){

        // случайные токи
        make_rand_external_current(
            N_neur_node,
            &I_ext_max,
            100,
            node_number,
            I_ext_node
        );

    }
    MPI_Allgatherv(I_ext_node, N_neur_node, MPI_FLOAT, I_ext, node_count_neur, node_displ_neur, MPI_FLOAT, MPI_COMM_WORLD);

#if DEBUG > 1
    if( node_number == 1 ){

        printf("\033[36m    Внешние токи, проверка на 1-м узле \033[30;0m");
        for ( i = 0; i < N_neur; ++i ){
            printf("%4.2lf ", I_ext[i]);
         }
        puts("");

    }
#endif

    // инициализация в начальный момент времени потенциала V и вспомогательной переменной u
    if( node_number ){

        for( i = 0; i < N_neur_node; ++i ){

             // определяем потенциал и вспомогательную переменную 
            V_node[i] = V_prev_node[i] = V_start; 
            u_node[i] = u_prev_node[i] = u_start;

        }
    }

    // сбор потенциала V_node и вспомогательной переменной u_node в единые вектора
    MPI_Allgatherv(V_node, N_neur_node, MPI_FLOAT, V, node_count_neur, node_displ_neur, MPI_FLOAT, MPI_COMM_WORLD);
    MPI_Allgatherv(u_node, N_neur_node, MPI_FLOAT, u, node_count_neur, node_displ_neur, MPI_FLOAT, MPI_COMM_WORLD);


    // инициализация в начальный момент времени синаптического тока I_syn 
    if( node_number ){

        for( i = 0; i < N_neur_node; ++i )
            I_syn_node[i] = I_syn_start;

    } // if node_number

    // сбор массивов синаптических токов в единый
    MPI_Allgatherv(I_syn_node, N_neur_node, MPI_FLOAT, I_syn, node_count_neur, node_displ_neur, MPI_FLOAT, MPI_COMM_WORLD);

    // инициализация вспомогательных модулирующих переменных y_con_*
    if( node_number ){
        for( i=0; i<N_con_node; ++i) { 
            y_con_prev_node[i] = y_con_curr_node[i] = y_con_start;
        }
    } // if node_number

#if DEBUG > 1
    MPI_Barrier(MPI_COMM_WORLD);
    if( !node_number ){

        printf("\033[36m    Начальный потенциал, проверка на управляющем узле: \033[30;0m");
        for( i = 0; i < N_neur; ++i ) {
            printf("%6.2lf ", V[i]);
        }
        puts("");

        printf("\033[36m    Начальное значение u, проверка на управляющем узле: \033[30;0m");
        for( i = 0; i < N_neur; ++i ) {
            printf("%6.2lf ", u[i]);
        }
        puts("");

        printf("\033[36m    Синаптические токи, проверка на управляющем узле \033[30;0m");
        for ( i = 0; i < N_neur; ++i ){
            printf("%4.2lf ", I_syn[i]);
         }
        puts("");


    }
    MPI_Barrier(MPI_COMM_WORLD);
    if( node_number == 1 ){

        printf("\033[36m    Начальный потенциал, проверка на 1-м узле:         \033[30;0m");
        for( i = 0; i < N_neur; ++i ) {
            printf("%6.2lf ", V[i]);
        }
        puts("");

        printf("\033[36m    Начальное значение u, проверка на 1-м узле: \033[30;0m");
        for( i = 0; i < N_neur; ++i ) {
            printf("%6.2lf ", u[i]);
        }
        puts("");

        printf("\033[36m    Синаптические токи, проверка на 1-м узле \033[30;0m");
        for ( i = 0; i < N_neur; ++i ){
            printf("%4.2lf ", I_syn[i]);
         }
        puts("");

    }
#endif




#if DEBUG > 0
    MPI_Barrier(MPI_COMM_WORLD);
    if( !node_number )
        puts("\033[1mПроведение вычислений \033[30;0m");
#endif

    // --- вычислительный процесс на подчиненных узлах

    float t = 0.;
    while( t <= t_end ) {

        // - выполняем решение СДУ на элементарном шаге
 
        // вычисление элементарного действия из дискретной схемы СДУ
        if( node_number ){

           #if DEBUG > 2
                printf("\033[30;1m    node %d: N_neur_node=%d  V_node[0]=%lf u_node[0]=%lf  I_syn[0]=%lf \
                I_ext[0]=%lf C=%lf V_a=%lf V_b=%lf V_c=%lf u_a=%lf ub=%lf *** \033[30;0m \n", 
               node_number, N_neur_node, V_node[0], u_node[0], I_syn[0], I_ext[0], C, V_a, V_b, V_c, u_a, u_b);
           #endif

           if( isGPU ){

               F_Ishikevich_GPU(
                   N_neur_node, 
                   V_node, 
                   u_node, 
                   I_syn_node, 
                   I_ext_node, 
                   &C, 
                   &V_a, 
                   &V_b, 
                   &V_c, 
                   &u_a, 
                   &u_b, 
                   k_V_node, 
                   k_u_node
               );

           }else{

               F_Ishikevich(
                   N_neur_node, 
                   V_node, 
                   u_node, 
                   I_syn_node, 
                   I_ext_node, 
                   &C, 
                   &V_a, 
                   &V_b, 
                   &V_c, 
                   &u_a, 
                   &u_b, 
                   k_V_node, 
                   k_u_node
               );

           }

           // временно метод Эйлера
           for( i = 0; i < N_neur_node; ++i ){
               V_node[i] = V_prev_node[i] + dt * k_V_node[i];
               u_node[i] = u_prev_node[i] + dt * k_u_node[i];
            }

           #if DEBUG > 1
               printf("\033[36m    Локальные значения потенциала на %d узле: \033[30;0m ", node_number);
               for( i = 0; i < N_neur_node; ++i ) {
                   printf("%4.2lf ", V_node[i]);
               }
               puts("");
               printf("\033[36m    Локальные значения вспомогательной переменной u на %d узле: \033[30;0m ", node_number);
               for( i = 0; i < N_neur_node; ++i ) {
                   printf("%4.2lf ", u_node[i]);
               }
               puts("");
           #endif

        } // if node_number

        // условие из СДУ для спайка, расположено вне правой части

        #pragma omp parallel for private(i) default(shared)
        for( i = 0; i < N_neur_node; ++i ){

            if( V_prev_node[i] > V_lim ) {

                 V_node[i] = V_after_spike;
                 u_node[i] = u_prev_node[i] + u_d;
             }
        }

        //#pragma omp parallel for private(i) default(shared)
        for( i = 0; i < N_neur_node; ++i ){

            if( V_prev_node[i] > V_lim ) {

                 int number_neur = 0;
                 for( j = 1; j < node_number; ++j ) {
                     number_neur += node_count_neur[j];
                 }

                 if( idx_spike_node == N_spike_node ){

                     N_spike_node *= 2;
                     neur_spike_node = realloc( neur_spike_node, N_spike_node*sizeof(int) );
                     t_spike_node = realloc( t_spike_node, N_spike_node*sizeof(float) );

                 }

                 neur_spike_node[idx_spike_node] = number_neur + i;
                 t_spike_node[idx_spike_node] = t;
                 ++idx_spike_node;

             } // if

        } // for i




        #pragma omp parallel for private(i) default(shared)
        for( i = 0; i < N_neur_node; ++i ){
            V_prev_node[i] = V_node[i];
            u_prev_node[i] = u_node[i];
        }


        // сбор потенциалов V_node и вспомогательной переменной u_node в единые вектора
        MPI_Allgatherv(V_node, N_neur_node, MPI_FLOAT, V, node_count_neur, node_displ_neur, MPI_FLOAT, MPI_COMM_WORLD);
        MPI_Allgatherv(u_node, N_neur_node, MPI_FLOAT, u, node_count_neur, node_displ_neur, MPI_FLOAT, MPI_COMM_WORLD);


        #if DEBUG > 1
        if(!node_number){
            printf("\033[36m    Потенциал V, проверка на управляющем узле: \033[30;0m");
            for( i = 0; i<N_neur; ++i) {
                printf("%4.2lf ", V[i]);
            }
            puts("");
        }
        #endif

        // перерасчет синаптического тока
        if( node_number ){
#if VIDEO < 1

            //убывание тока по экспоненте со скачком при спайке
            I_synaptic_exp(
                N_neur,
                V,
                &dt, 
                &tau,
                &V_lim,
                N_con_node,
                y_con_prev_node,
                pre_con_node,
                post_con_node,
                weights_con_node,
                part_I_syn_node, // используется 0..N_neur
                y_con_curr_node
            );

#endif
#if VIDEO > 0
            I_synaptic_exp_gpu(
        	    N_neur,
        	    V,
        	    dt,
        	    tau,
        	    V_lim,
        	    N_con_node,
        	    y_con_prev_node,
        	    pre_con_node,
        	    post_con_node,
        	    weights_con_node,
        	    part_I_syn_node, // используется 0..N_neur
        	    y_con_curr_node,
        	    SIZE_BLOCK_X,
        	    SIZE_BLOCK_Y
        	);
#endif
            #pragma omp parallel for private(i) default(shared)
            for( i=0; i<N_con_node; ++i)
                y_con_prev_node[i] = y_con_curr_node[i];

            #if DEBUG > 2
                printf("\033[30;1m    node %d: N_neur=%d  V_node[0]=%f dt=%f tau=%f V_lim=%f N_con_node=%d \
                y_con_prev_node[0]=%f pre_con_node[0]=%d post_con_node[0]=%d weights_con_node[0]=%f \
                part_I_syn_node[0]=%f y_con_curr_node[0]=%f \033[30;0m \n", 
                node_number, N_neur, V_node[0], dt, tau, V_lim, N_con_node, y_con_prev_node[0], pre_con_node[0], 
                post_con_node[0], weights_con_node[0], part_I_syn_node[0], y_con_curr_node[0]);
            #endif

        } else {
            #pragma omp parallel for private(i) default(shared)
            for( i=0; i<N_neur; ++i )
                part_I_syn_node[i] = 0.;
        }

        // сбор массивов синаптических токов в единый
        MPI_Allreduce( part_I_syn_node, I_syn, N_neur, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD );

        #if DEBUG > 1
            if( ! node_number ){
                printf("\033[36m    Синаптический ток на управляющем узле: \033[30;0m");
                for( i=0; i<N_neur; ++i) {
                    printf("%8.6lf ", I_syn[i]);
                }
                puts("");
            }
        #endif

        // Значения из массива для синаптического тока делим по узлам
        MPI_Scatterv(I_syn, node_count_neur, node_displ_neur, MPI_FLOAT, I_syn_node, N_neur_node,  MPI_FLOAT, 0, MPI_COMM_WORLD  );

        #if DEBUG > 1
            if( node_number == 1 ){
                printf("\033[36m    Порция значений синаптического тока на первом узле: \033[30;0m");
                for(i=0;i<N_neur_node;++i) 
                    printf("%8.6lf ",I_syn_node[i]);
                puts("");
            }
        #endif


        // запись результатов пока в файл

        t += dt;
    } // while time


#if DEBUG > 0
    MPI_Barrier(MPI_COMM_WORLD);
    if( !node_number )
        puts("\033[1mСохранение результатов \033[30;0m");
#endif


    // --- собираем спайки в единый массив

    // вычисляем число спайков
    MPI_Reduce( &idx_spike_node, &N_spike, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD );

    // выделяем память на нулевом узле
    if( !node_number ) {

        neur_spike_znode = (int*)malloc( N_spike*sizeof(int) );
        if( !neur_spike_znode ){
            puts("\033[31;1mОшибка выделения памяти (neur_spike_node) на нулевом узле \033[30;0m ");
            MPI_Abort(MPI_COMM_WORLD,1);
        }

        t_spike_znode = (float*)malloc( N_spike*sizeof(float) );
        if( !t_spike_znode ){
            puts("\033[31;1mОшибка выделения памяти (t_spike_node) на нулевом узле\033[30;0m ");
            MPI_Abort(MPI_COMM_WORLD,1);
        }

    } // if !node_number


    int* node_displ_spike = (int*)malloc(nodes_number*sizeof(int));
    if( !node_displ_spike ){
        puts("\033[31;1mОшибка выделения памяти (node_displ_spike) \033[30;0m ");
        MPI_Abort(MPI_COMM_WORLD,1);
    }

    int* node_count_spike = (int*)malloc(nodes_number*sizeof(int));
    if( !node_count_spike ){
        puts("\033[31;1mОшибка выделения памяти (node_count_spike) \033[30;0m ");
        MPI_Abort(MPI_COMM_WORLD,1);
    }

    MPI_Allgather( &idx_spike_node, 1, MPI_INT, node_count_spike, 1, MPI_INT, MPI_COMM_WORLD );

    node_displ_spike[0] = 0;
    #pragma omp parallel for private(i) default(shared)
    for( i = 1; i < nodes_number; ++i ){
        node_displ_spike[i] = 0;
        for( j=0; j<i; ++j ) 
            node_displ_spike[i] += node_count_spike[j];
    }

    // собираем все спайки на нулевом узле
    MPI_Gatherv(neur_spike_node, idx_spike_node, MPI_INT, neur_spike_znode, node_count_spike, node_displ_spike, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Gatherv(t_spike_node, idx_spike_node, MPI_INT, t_spike_znode, node_count_spike, node_displ_spike, MPI_INT, 0, MPI_COMM_WORLD);

    // временно для тестов так
    if( !node_number ) {
        FILE *f_spike = fopen("spikes.dat", "w+");
        for( i = 0; i < N_spike; ++i )
            fprintf( f_spike,"%6.2lf %6i\n", t_spike_znode[i], neur_spike_znode[i] );
        fclose(f_spike); 
    }
    //for( i = 0; i < N_spike; ++i ) 
    //    printf("%8.2f %d\n",t_spike_znode[i],neur_spike_znode[i]);

    // удаляем память для массивов, учавствующих в расчетах
    #include "free_add_mass.c"

    // удаляем массивы, учавствующие в распределении нагрузки кластера
    free(node_displ_neur);
    free(node_count_neur);
    free(node_displ_con);
    free(node_count_con);

    MPI_Finalize();


    return 0;
}
