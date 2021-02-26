//#include "./gaussian_kernel.h" 
//#include "device_launch_parameters.h"
//
//#define BLOCK 32.0
//#define TILE_WIDTH 32
//
///*
//The actual gaussian blur kernel to be implemented by 
//you. Keep in mind that the kernel operates on a 
//single channel.
// */
//__global__ 
//void gaussianBlur(unsigned char *d_in, unsigned char *d_out, 
//        const int rows, const int cols, float *d_filter, const int filterWidth){
//
//	__shared__ unsigned char ds_M[TILE_WIDTH][TILE_WIDTH];
//	__shared__ int p;
//
//	int x = blockIdx.x*blockDim.x + threadIdx.x;
//	int y = blockIdx.y*blockDim.y + threadIdx.y;
//	float pixVal = 0;
//
//	// Load data to the shared memory
//	// Boundary check
//	p = x / TILE_WIDTH;
//	if ((y < rows) && (p*TILE_WIDTH + threadIdx.x < cols)) {
//		ds_M[threadIdx.y][threadIdx.x] = d_in[y*cols + p*TILE_WIDTH + threadIdx.x];
//	}
//	else {
//		ds_M[threadIdx.y][threadIdx.x] = 0;
//	}
//
//	// Barrier synchronization
//	__syncthreads();
//
//	// Check the boundary condition
//	p = x / TILE_WIDTH;
//	if ((y < rows) && (p*TILE_WIDTH + threadIdx.x < cols)) {
//
//		// Reset parameters
//		pixVal = 0;
//
//		// Obtain pixels right under the kernel
//		for (int blur_r = -(filterWidth / 2); blur_r <= (filterWidth / 2); blur_r++) {
//			for (int blur_c = -(filterWidth / 2); blur_c <= (filterWidth / 2); blur_c++) {
//
//				// Calculate the index of the current row and that of the current column
//				int current_row = y + blur_r;
//				int current_col = x + blur_c;
//
//				// Boundary check
//				if ((current_row >= 0) && (current_row < rows) && (current_col >= 0) && (current_col < cols)) {
//					pixVal += d_in[current_row*cols + current_col] * d_filter[(blur_r + filterWidth / 2)*filterWidth + (blur_c + filterWidth / 2)];
//				}
//
//			}
//		}
//
//		// Save the result
//		d_out[y*cols + x]= unsigned char(pixVal);
//	}
//
//} 
//
//
//
///*
//  Given an input RGBA image separate 
//  that into appropriate rgba channels.
// */
//__global__ 
//void separateChannels(uchar4 *d_imrgba, unsigned char *d_r, unsigned char *d_g, unsigned char *d_b,
//        const int rows, const int cols){
//	int x = blockIdx.x*blockDim.x + threadIdx.x;
//	int y = blockIdx.y*blockDim.y + threadIdx.y;
//	int ind = 0;
//
//	// Check the boundary
//	if ((x < cols) && (y < rows)) {
//		ind = y*cols + x;
//		
//		// Separate the channels
//		d_r[ind] = d_imrgba[ind].x;
//		d_g[ind] = d_imrgba[ind].y;
//		d_b[ind] = d_imrgba[ind].z;
//	}
//} 
// 
//
///*
//  Given input channels combine them 
//  into a single uchar4 channel. 
//
//  You can use some handy constructors provided by the 
//  cuda library i.e. 
//  make_int2(x, y) -> creates a vector of type int2 having x,y components 
//  make_uchar4(x,y,z,255) -> creates a vector of uchar4 type x,y,z components 
//  the last argument being the transperency value. 
// */
//__global__ 
//void recombineChannels(unsigned char *d_r, unsigned char *d_g, unsigned char *d_b, uchar4 *d_orgba,
//        const int rows, const int cols){
//	int x = blockIdx.x*blockDim.x + threadIdx.x;
//	int y = blockIdx.y*blockDim.y + threadIdx.y;
//	int ind = 0;
//
//	// Check the boundary
//	if ((x < cols) && (y < rows)) {
//		ind = y*cols + x;
//
//		// Combine the three channels
//		d_orgba[ind].x = d_r[ind];
//		d_orgba[ind].y = d_g[ind];
//		d_orgba[ind].z = d_b[ind];
//		d_orgba[ind].w = 255;
//	}
//} 
//
//
//void your_gauss_blur(uchar4* d_imrgba, uchar4 *d_oimrgba, size_t rows, size_t cols, 
//        unsigned char *d_red, unsigned char *d_green, unsigned char *d_blue, 
//        unsigned char *d_rblurred, unsigned char *d_gblurred, unsigned char *d_bblurred,
//        float *d_filter,  int filterWidth){
// 
//
//	dim3 gridSize(int(ceil(cols / BLOCK)), int(ceil(rows / BLOCK)), 1);
//    dim3 blockSize(int(BLOCK), int(BLOCK), 1);
//
//    separateChannels<<<gridSize, blockSize>>>(d_imrgba, d_red, d_green, d_blue, int(rows), int(cols));
//    cudaDeviceSynchronize();
//    checkCudaErrors(cudaGetLastError());
//
//    gaussianBlur<<<gridSize, blockSize>>>(d_red, d_rblurred, int(rows), int(cols), d_filter, filterWidth);
//    cudaDeviceSynchronize();
//    checkCudaErrors(cudaGetLastError());
//
//    gaussianBlur<<<gridSize, blockSize>>>(d_green, d_gblurred, int(rows), int(cols), d_filter, filterWidth);
//    cudaDeviceSynchronize();
//    checkCudaErrors(cudaGetLastError());
//
//    gaussianBlur<<<gridSize, blockSize>>>(d_blue, d_bblurred, int(rows), int(cols), d_filter, filterWidth);
//    cudaDeviceSynchronize();
//    checkCudaErrors(cudaGetLastError());
//
//    recombineChannels<<<gridSize, blockSize>>>(d_rblurred, d_gblurred, d_bblurred, d_oimrgba, int(rows), int(cols));
//    cudaDeviceSynchronize();
//    checkCudaErrors(cudaGetLastError());   
//
//}




