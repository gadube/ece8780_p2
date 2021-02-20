#include "./gaussian_kernel.h" 

#ifndef BLOCK
#define BLOCK 32
#endif

/*
The actual gaussian blur kernel to be implemented by 
you. Keep in mind that the kernel operates on a 
single channel.
 */
__global__ 
void gaussianBlur(unsigned char *d_in, unsigned char *d_out, 
        const int rows, const int cols, float *d_filter, const int filterWidth){
	int pixVal;
	int start_col, start_row;
	int curr_col, curr_row;
	int c = blockIdx.x * blockDim.x + threadIdx.x;
	int r = blockIdx.y * blockDim.y + threadIdx.y;

	if (c < cols && r < rows){
		pixVal = 0;

		start_col = c - (filterWidth / 2);
		start_row = r - (filterWidth / 2);
		
		for (int i = 0; i < filterWidth; ++i){
			for (int j = 0; j < filterWidth; ++j){
				curr_row = start_row + i;
				curr_col = start_col + j;

				if (curr_row > -1 && curr_row < rows && curr_col > -1 && curr_col < cols){
					pixVal += d_in[curr_row * cols + curr_col] * d_filter[i * filterWidth + j];
				}
			}
		}

		d_out[r * cols + c] = (unsigned char) pixVal;
	}

	return;
} 


/*
  Given an input RGBA image separate 
  that into appropriate rgba channels.
 */
__global__ 
void separateChannels(uchar4 *d_imrgba, unsigned char *d_r, unsigned char *d_g, unsigned char *d_b,
        const int rows, const int cols){
	int i;
	int c = blockIdx.x * blockDim.x + threadIdx.x;
	int r = blockIdx.y * blockDim.y + threadIdx.y;

	// split uchar4 into r, g, b px values
	if (c < cols && r < rows){
		i = r * cols + c;
		d_r[i] = d_imrgba[i].x;
		d_g[i] = d_imrgba[i].y;
		d_b[i] = d_imrgba[i].z;
	}
	
	return;
} 
 

/*
  Given input channels combine them 
  into a single uchar4 channel. 

  You can use some handy constructors provided by the 
  cuda library i.e. 
  make_int2(x, y) -> creates a vector of type int2 having x,y components 
  make_uchar4(x,y,z,255) -> creates a vector of uchar4 type x,y,z components 
  the last argument being the transperency value. 
 */
__global__ 
void recombineChannels(unsigned char *d_r, unsigned char *d_g, unsigned char *d_b, uchar4 *d_orgba,
        const int rows, const int cols){

	int i;
	int c = blockIdx.x * blockDim.x + threadIdx.x;
	int r = blockIdx.y * blockDim.y + threadIdx.y;
	
	// recombine R,G, and B px values int uchar4
	if (c < cols && r < rows){
		i = r * cols + c;
		d_orgba[i] = make_uchar4(d_r[i], d_g[i], d_b[i], 255);
	}
	
	return;
} 


void your_gauss_blur(uchar4* d_imrgba, uchar4 *d_oimrgba, size_t rows, size_t cols, 
        unsigned char *d_red, unsigned char *d_green, unsigned char *d_blue, 
        unsigned char *d_rblurred, unsigned char *d_gblurred, unsigned char *d_bblurred,
        float *d_filter,  int filterWidth){
 


        dim3 blockSize(BLOCK,BLOCK,1);
        dim3 gridSize((cols + BLOCK - 1)/BLOCK,(rows + BLOCK - 1)/BLOCK,1);

        separateChannels<<<gridSize, blockSize>>>(d_imrgba, d_red, d_green, d_blue, rows, cols);
        cudaDeviceSynchronize();
        checkCudaErrors(cudaGetLastError());

        gaussianBlur<<<gridSize, blockSize>>>(d_red, d_rblurred, rows, cols, d_filter, filterWidth);
        cudaDeviceSynchronize();
        checkCudaErrors(cudaGetLastError());

        gaussianBlur<<<gridSize, blockSize>>>(d_green, d_gblurred, rows, cols, d_filter, filterWidth);
        cudaDeviceSynchronize();
        checkCudaErrors(cudaGetLastError());

        gaussianBlur<<<gridSize, blockSize>>>(d_blue, d_bblurred, rows, cols, d_filter, filterWidth);
        cudaDeviceSynchronize();
        checkCudaErrors(cudaGetLastError());

        recombineChannels<<<gridSize, blockSize>>>(d_rblurred, d_gblurred, d_bblurred, d_oimrgba, rows, cols);

        cudaDeviceSynchronize();
        checkCudaErrors(cudaGetLastError());   

}




