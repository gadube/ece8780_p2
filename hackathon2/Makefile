NVCC=nvcc 

OPENCV_INCLUDE_PATH="$(OPENCV_ROOT)/include/opencv4"

OPENCV_LD_FLAGS = -L $(OPENCV_ROOT)/lib64 -lopencv_core -lopencv_imgproc -lopencv_imgcodecs

CUDA_INCLUDEPATH=/usr/local/cuda/include

NVCC_OPTS=-arch=sm_70 -DBLOCK=32
GCC_OPTS=-std=c++11 -g -O3 -Wall 
CUDA_LD_FLAGS=-L $(CUDA_ROOT)/lib64 -lcudart

final: main.o blur.o
	g++ -o gblur main.o Gaussian_filter_example.o $(CUDA_LD_FLAGS) $(OPENCV_LD_FLAGS)

main.o:main.cpp gaussian_kernel.h utils.h 
	g++ -c $(GCC_OPTS) -I $(CUDA_INCLUDEPATH) -I $(OPENCV_INCLUDE_PATH) main.cpp

blur.o:  Gaussian_filter_example.cu gaussian_kernel.h  utils.h
	$(NVCC) -c Gaussian_filter_example.cu $(NVCC_OPTS)

clean:
	rm -rf *.o gblur blurred_gpu.png blurred_serial.png
