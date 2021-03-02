#include "./gaussian_kernel.h" 
#include "device_launch_parameters.h"

#ifndef BLOCK
#define BLOCK 32
#endif


// The kernel function for image filtering using shared memory
__global__ void imageFilteringKernel(const unsigned char *d_f, const unsigned int blockW, const unsigned int blockH, const int S,
	unsigned char *d_h, const unsigned int W, const unsigned int H, float *d_cFilterKernel)
{

	float pixVal = 0;

	// Set the size of a tile (Note: tileW = tileH)
	const unsigned int tileW = blockW + 2 * S;

	// Set the start position of the block
	// Note that the origin of the first block is ( S, S )
	const unsigned int blockStartCol = blockIdx.x * blockW + S;
	const unsigned int blockEndCol = blockStartCol + blockW;
	const unsigned int blockStartRow = blockIdx.y * blockH + S;
	const unsigned int blockEndRow = blockStartRow + blockH;

	// Set the position of the tile
	// Note that the origin of the first tile is(0, 0)
	const unsigned int tileStartCol = blockStartCol - S;
	const unsigned int tileEndCol = blockEndCol + S;
	const unsigned int tileStartRow = blockStartRow - S;
	const unsigned int tileEndRow = blockEndRow + S;

	// Make sure the tile is within the padded image
	const unsigned int tileEndClampedCol = ((tileEndCol<W) ? tileEndCol : W);
	const unsigned int tileEndClampedRow = ((tileEndRow<H) ? tileEndRow : H);

	// Set the size of the filter kernel
	// Note: kernelSize denotes the width (or height) of the kernel
	const unsigned int kernelSize = 2 * S + 1;

	// The shared memory for the tile
	extern __shared__ unsigned char sData[];

	// Copy the tile into shared memory
	// Recall that the block size is (32,32)
	unsigned int tilePixelPosCol = threadIdx.x;
	unsigned int iPixelPosCol = tileStartCol + threadIdx.x;
	unsigned int tilePixelPosRow = threadIdx.y;
	unsigned int iPixelPosRow = tileStartRow + threadIdx.y;
	// Check if the pixel is in the image
	if (iPixelPosCol < tileEndClampedCol && iPixelPosRow < tileEndClampedRow) { 
			unsigned int iPixelPos = iPixelPosRow * W + iPixelPosCol;
			unsigned int tilePixelPos = tilePixelPosRow * tileW + tilePixelPosCol;
			sData[tilePixelPos] = d_f[iPixelPos];
	}

	// Wait for all the threads for data loading
	// After '__syncthreads();', the shared memory contains 32x32 pixels
	__syncthreads();

	// Perform convolution
	// Reset the x and y coordinate
	tilePixelPosCol = threadIdx.x;
	iPixelPosCol = tileStartCol + tilePixelPosCol;
	tilePixelPosRow = threadIdx.y;
	iPixelPosRow = tileStartRow + threadIdx.y;

	// Check if the pixel is in the tile and image.
	if(iPixelPosCol >= tileStartCol && iPixelPosCol < tileEndClampedCol &&
			iPixelPosRow >= tileStartRow && iPixelPosRow < tileEndClampedRow){ 

			// Compute the pixel position for the output image
			unsigned int oPixelPosCol = iPixelPosCol;
			unsigned int oPixelPosRow = iPixelPosRow;
			unsigned int oPixelPos = oPixelPosRow * W + oPixelPosCol;

			// Compute the pixel position within the tile
			unsigned int tilePixelPos = tilePixelPosRow * tileW + tilePixelPosCol;

			// Ignore the pixels on the edge of the images
			if (iPixelPosCol >= tileStartCol + S && iPixelPosCol < tileEndClampedCol - S &&
				iPixelPosRow >= tileStartRow + S && iPixelPosRow < tileEndClampedRow - S) {

				// Reset parameters
				pixVal = 0;

				// Apply Gaussian filter to the image
				d_h[oPixelPos] = 0.0;
				for (int i = -S; i <= S; i++) {
					for (int j = -S; j <= S; j++) {
						int tilePixelPosOffset = i * tileW + j;
						int coefPos = (i + S) * kernelSize + (j + S); 
						pixVal += sData[tilePixelPos + tilePixelPosOffset] * d_cFilterKernel[coefPos];
					}
				}

				// Save the result
				d_h[oPixelPos] = int(pixVal);
			}
			// Save the pixels for the left and the right edge
			else if(iPixelPosCol<S || iPixelPosCol>=(W-S)){
				d_h[oPixelPos] = sData[tilePixelPos];
			}
			// Save the pixels for the top and the bottom edge
			else if(iPixelPosRow<S || iPixelPosRow>=(H-S))
				d_h[oPixelPos] = sData[tilePixelPos];
		}

}

  //Given an input RGBA image separate 
  //that into appropriate rgba channels.
__global__ 
void separateChannels(uchar4 *d_imrgba, unsigned char *d_r, unsigned char *d_g, unsigned char *d_b,
        const int rows, const int cols){
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
	int ind = 0;

	// Check the boundary
	if ((x < cols) && (y < rows)) {
		ind = y*cols + x;
		
		// Separate the channels
		d_r[ind] = d_imrgba[ind].x;
		d_g[ind] = d_imrgba[ind].y;
		d_b[ind] = d_imrgba[ind].z;
	}
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
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
	int ind = 0;

	// Check the boundary
	if ((x < cols) && (y < rows)) {
		ind = y*cols + x;

		// Combine the three channels
		d_orgba[ind].x = d_r[ind];
		d_orgba[ind].y = d_g[ind];
		d_orgba[ind].z = d_b[ind];
		d_orgba[ind].w = 255;
	}
} 


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


__global__ 
void gaussianBlur_col_separable(unsigned char *d_in, unsigned char *d_out, 
        const int rows, const int cols, float *d_filter, const int filter_width){

	int pixVal;
	int current_row;
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	if (j < cols && i < rows){
		// Reset parameters
		pixVal = 0;

		// Obtain pixels right under the kernel
		for (int blur_c = -(filter_width / 2); blur_c <= (filter_width / 2); blur_c++) {

			// Calculate the index of the current row and that of the current column
			current_row = i + blur_c;

			// Boundary check
			if ((current_row >= 0) && (current_row < rows)) {
				pixVal += d_in[current_row*cols + j]*d_filter[(filter_width / 2) - blur_c];
			}
		}
	
		// Save the result
		d_out[i*cols + j] = (unsigned char)(pixVal);
		//d_out[i*cols + j] = d_in[i*cols + j];

	}	
	return;
} 

__global__ 
void gaussianBlur_row_separable(unsigned char *d_in, unsigned char *d_out, 
        const int rows, const int cols, float *d_filter, const int filter_width){

	int pixVal;
	int current_col;
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < rows && j < cols){

		// Reset parameters
		pixVal = 0;

		// Obtain pixels right under the kernel
		for (int blur_r = -(filter_width / 2); blur_r <= (filter_width / 2); blur_r++) {

		// Calculate the index of the current row and that of the current column
			current_col = j + blur_r;

			// Boundary check
			if ((current_col >= 0) && (current_col < cols)) {
				pixVal += d_in[i*cols + current_col]*d_filter[(filter_width / 2) - blur_r];
			}
		}

		d_out[i * cols + j] = (unsigned char) pixVal;
	}

	return;
} 

void gauss_blur_shared_mem(uchar4 *d_padded_image, float *d_filter, int filter_width, uchar4 *d_filterred_result, int img_width, int img_height,
	unsigned char *d_red, unsigned char *d_green, unsigned char *d_blue,
	unsigned char *d_rblurred, unsigned char *d_gblurred, unsigned char *d_bblurred)
{
	// Set the execution configuration for the Gaussian filter kernel
	const unsigned int tileW = BLOCK;
	const unsigned int tileH = BLOCK;
	const unsigned int blockW = tileW - 2*4;
	const unsigned int blockH = tileH - 2*4;

	// Set the size of shared memory
	const unsigned int sharedMemorySizeByte = tileW * tileH * sizeof(float);

	// Separate the channels
	dim3 gridSize_padded(int(ceil(img_width / BLOCK) + 1), int(ceil(img_height / BLOCK) + 1), 1);
	dim3 blockSize_padded(int(BLOCK), int(BLOCK), 1);
	separateChannels <<<gridSize_padded, blockSize_padded>>>(d_padded_image, d_red, d_green, d_blue, int(img_height), int(img_width));
	cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());

	// Apply Gaussian filter to the red channel
	const dim3 grid(int(ceil(img_width / blockW) + 1), int(ceil(img_height / blockH) + 1));
	const dim3 threadBlock(tileW, tileH);
	imageFilteringKernel <<<grid, threadBlock, sharedMemorySizeByte>>>(d_red, blockW, blockH, 4,
		d_rblurred, img_width, img_height, d_filter);
	cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());

	// Apply Gaussian filter to the green channel
	imageFilteringKernel <<<grid, threadBlock, sharedMemorySizeByte>>>(d_green, blockW, blockH, 4,
		d_gblurred, img_width, img_height, d_filter);
	cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());

	// Apply Gaussian filter to the blue channel
	imageFilteringKernel <<<grid, threadBlock, sharedMemorySizeByte>>>(d_blue, blockW, blockH, 4,
		d_bblurred, img_width, img_height, d_filter);
	cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());

	// Recombine the channels
	dim3 gridSize_recombine(int(ceil(img_width / BLOCK) + 1), int(ceil(img_height / BLOCK) + 1), 1);
	dim3 blockSize_recombine(int(BLOCK), int(BLOCK), 1);
	recombineChannels <<<gridSize_recombine, blockSize_recombine>>>(d_rblurred, d_gblurred, d_bblurred, d_filterred_result, int(img_height), int(img_width));
	cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());
}

void separable_gauss_blur(uchar4* d_imrgba, uchar4 *d_oimrgba, size_t rows, size_t cols, 
        unsigned char *d_red, unsigned char *d_green, unsigned char *d_blue, 
        unsigned char *d_rblurred, unsigned char *d_gblurred, unsigned char *d_bblurred,
        float *d_row_filter, float *d_col_filter,  int filterWidth){

        dim3 blockSize(BLOCK,BLOCK,1);
        dim3 gridSize((cols + BLOCK - 1)/BLOCK,(rows + BLOCK - 1)/BLOCK,1);

        separateChannels<<<gridSize, blockSize>>>(d_imrgba, d_red, d_green, d_blue, rows, cols);
        cudaDeviceSynchronize();
        checkCudaErrors(cudaGetLastError());

				//blur red channel
        gaussianBlur_col_separable<<<gridSize, blockSize>>>(d_red, d_rblurred, rows, cols, d_col_filter, filterWidth);
        cudaDeviceSynchronize();
        checkCudaErrors(cudaGetLastError());

        gaussianBlur_row_separable<<<gridSize, blockSize>>>(d_rblurred, d_red, rows, cols, d_row_filter, filterWidth);
        cudaDeviceSynchronize();
        checkCudaErrors(cudaGetLastError());

				//blur green channel
        gaussianBlur_col_separable<<<gridSize, blockSize>>>(d_green, d_gblurred, rows, cols, d_col_filter, filterWidth);
        cudaDeviceSynchronize();
        checkCudaErrors(cudaGetLastError());

        gaussianBlur_row_separable<<<gridSize, blockSize>>>(d_gblurred, d_green, rows, cols, d_row_filter, filterWidth);
        cudaDeviceSynchronize();
        checkCudaErrors(cudaGetLastError());

				//blur blue channel
        gaussianBlur_col_separable<<<gridSize, blockSize>>>(d_blue, d_bblurred, rows, cols, d_col_filter, filterWidth);
        cudaDeviceSynchronize();
        checkCudaErrors(cudaGetLastError());

        gaussianBlur_row_separable<<<gridSize, blockSize>>>(d_bblurred, d_blue, rows, cols, d_row_filter, filterWidth);
        cudaDeviceSynchronize();
        checkCudaErrors(cudaGetLastError());

        recombineChannels<<<gridSize, blockSize>>>(d_red, d_green, d_blue, d_oimrgba, rows, cols);

        cudaDeviceSynchronize();
        checkCudaErrors(cudaGetLastError());   

}

void original_gauss_blur(uchar4* d_imrgba, uchar4 *d_oimrgba, size_t rows, size_t cols, 
        unsigned char *d_red, unsigned char *d_green, unsigned char *d_blue, 
        unsigned char *d_rblurred, unsigned char *d_gblurred, unsigned char *d_bblurred,
        float *d_filter,  int filterWidth){
 


        dim3 blockSize(BLOCK,BLOCK,1);
        dim3 gridSize((cols + BLOCK - 1)/BLOCK + 1,(rows + BLOCK - 1)/BLOCK + 1,1);

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
