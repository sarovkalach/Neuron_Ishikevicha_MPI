CXX = mpicc
LIB = -lm -fopenmp
OBJ = main.o par_utils.o models_neurons.o models_connections.o ode_numerical.o
FLG = -std=c99 -g -O0 -fopenmp 

main: $(OBJ)
	$(CXX) $^ -o $@ $(LIB)

main.o: main.c
	$(CXX) -c $^ $(LIB) $(FLG) -o $@

par_utils.o: par_utils.c par_utils.h
	$(CXX) -c $(FLG) $< -o $@

models_neurons.o: models_neurons.c models_neurons.h
	$(CXX) -c $(FLG) $< -o $@

models_connections.o: models_connections.c models_connections.h
	$(CXX) -c $(FLG) $< -o $@  

ode_numerical.o: ode_numerical.c ode_numerical.h
	$(CXX) -c $(FLG) $< -o $@  

clean:
	rm -f *.o main
