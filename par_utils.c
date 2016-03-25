#include "par_utils.h"


void calculate_portions(
    int  _nodes_number, 
    int  _actions_amount,
    int *_node_count,
    int *_node_map,
    int  isZeroCalc 
)
{

    if( !isZeroCalc ){

        int __chunk = _actions_amount / ( _nodes_number - 1 );

        int __remainder = _actions_amount % ( _nodes_number - 1 );

        _node_count[0]=0;
        for(int i=1; i<_nodes_number; ++i)
            _node_count[i] = i<=__remainder ? __chunk+1 : __chunk;

        _node_map[0] = 0;
        _node_map[1] = 0;
        for(int i=2; i<_nodes_number; ++i)
        _node_map[i] = _node_map[i-1] + _node_count[i-1];

    } else {

            // если потребуется 
            // добавить случай вычислений и на нулевом узле
    }
}

