#ifndef PAR_UTILS_NDN_2016
#define PAR_UTILS_NDN_2016

// функция выполняет распределение действий по вычислительным узлам 
// (можно использовать для разделения на любые потоки и процессы) 
// (chunk_map и chunk_count как в Scatterv)
void calculate_portions(
    int  _nodes_number, // число узлов
    int  _actions_amount, // всего действий
    int *_node_count,
    int *_node_map,
    int isZeroCalc // = 0 --- нулевой узел не получает порции действий
);

#endif
