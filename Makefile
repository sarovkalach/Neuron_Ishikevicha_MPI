CXX = mpicc
LIB = -lm -fopenmp
OBJ = main.o par_utils.o models_neurons.o models_connections.o ode_numerical.o models_neurons_gpu.o
FLG = -g -O0 -fopenmp 

main: $(OBJ)
	$(CXX) $^ -o $@ $(LIB) -L/usr/local/cuda-7.5/lib64 -lcuda -lcudart

main.o: main.c
	$(CXX) -c $^ $(LIB) $(FLG) -o $@

par_utils.o: par_utils.c par_utils.h
	$(CXX) -c $(FLG) $< -o $@

models_neurons.o: models_neurons.c models_neurons.h
	$(CXX) -c $(FLG) $< -o $@

models_neurons_gpu.o: models_neurons_gpu.cu models_neurons_gpu.h
	nvcc -ccbin gcc -c $< -o $@

models_connections.o: models_connections.c models_connections.h
	$(CXX) -c $(FLG) $< -o $@  

ode_numerical.o: ode_numerical.c ode_numerical.h
	$(CXX) -c $(FLG) $< -o $@  

clean:
	rm -f *.o main
