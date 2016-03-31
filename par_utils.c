#include "par_utils.h"


void calculate_portions(
    int  _nodes_number, 
    int  _actions_amount,
    int *_node_counts,
    int *_node_displs,
    int  isZeroCalc 
)
{
    int __i;

    if( !isZeroCalc ){

        int __count = _actions_amount / ( _nodes_number - 1 );

        int __remainder = _actions_amount % ( _nodes_number - 1 );

        _node_counts[0]=0;
        for( __i=1; __i<_nodes_number; ++__i)
            _node_counts[__i] = __i<=__remainder ? __count+1 : __count;

        _node_displs[0] =  _node_displs[1] = 0;
        for( __i=2; __i<_nodes_number; ++__i)
        _node_displs[__i] = _node_displs[__i-1] + _node_counts[__i-1];

    } else {

            // если потребуется 
            // добавить случай вычислений и на нулевом узле
    }
}

