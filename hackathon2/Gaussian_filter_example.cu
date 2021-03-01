#include "./gaussian_kernel.h" 
#include "device_launch_parameters.h"

///
/// The kernel function for image filtering using constant and shared memory
/// Note that passing references cannot be used.
///
template <typename T>
__global__ void imageFilteringKernel(const T *d_f, const unsigned int paddedW, const unsigned int paddedH,
	const unsigned int blockW, const unsigned int blockH, const int S,
	T *d_h, const unsigned int W, const unsigned int H, float *d_cFilterKernel)
{

	//
	// Note that blockDim.(x,y) cannot be used instead of blockW and blockH,
	// because the size of a thread block is not equal to the size of a data block
	// due to the apron and the use of subblocks.
	//

	//
	// Set the size of a tile
	//
	const unsigned int tileW = blockW + 2 * S;
	const unsigned int tileH = blockH + 2 * S;

	// 
	// Set the number of subblocks in a tile
	//
	const unsigned int noSubBlocks = static_cast<unsigned int>(ceil(static_cast<double>(tileH) / static_cast<double>(blockDim.y)));

	//
	// Set the start position of the block, which is determined by blockIdx. 
	// Note that since padding is applied to the input image, the origin of the block is ( S, S )
	//
	const unsigned int blockStartCol = blockIdx.x * blockW + S;
	const unsigned int blockEndCol = blockStartCol + blockW;

	const unsigned int blockStartRow = blockIdx.y * blockH + S;
	const unsigned int blockEndRow = blockStartRow + blockH;

	//
	// Set the position of the tile which includes the data block and its apron
	//
	const unsigned int tileStartCol = blockStartCol - S;
	const unsigned int tileEndCol = blockEndCol + S;
	const unsigned int tileEndClampedCol = min(tileEndCol, paddedW);

	const unsigned int tileStartRow = blockStartRow - S;
	const unsigned int tileEndRow = blockEndRow + S;
	const unsigned int tileEndClampedRow = min(tileEndRow, paddedH);

	//
	// Set the size of the filter kernel
	//
	const unsigned int kernelSize = 2 * S + 1;

	//
	// Shared memory for the tile
	//
	extern __shared__ T sData[];

	//
	// Copy the tile into shared memory
	//
	unsigned int tilePixelPosCol = threadIdx.x;
	unsigned int iPixelPosCol = tileStartCol + tilePixelPosCol;
	for (unsigned int subBlockNo = 0; subBlockNo < noSubBlocks; subBlockNo++) {

		unsigned int tilePixelPosRow = threadIdx.y + subBlockNo * blockDim.y;
		unsigned int iPixelPosRow = tileStartRow + tilePixelPosRow;

		if (iPixelPosCol < tileEndClampedCol && iPixelPosRow < tileEndClampedRow) { // Check if the pixel in the image
			unsigned int iPixelPos = iPixelPosRow * paddedW + iPixelPosCol;
			unsigned int tilePixelPos = tilePixelPosRow * tileW + tilePixelPosCol;
			sData[tilePixelPos] = d_f[iPixelPos];
		}

	}

	//
	// Wait for all the threads for data loading
	//
	__syncthreads();

	//
	// Perform convolution
	//
	tilePixelPosCol = threadIdx.x;
	iPixelPosCol = tileStartCol + tilePixelPosCol;
	for (unsigned int subBlockNo = 0; subBlockNo < noSubBlocks; subBlockNo++) {

		unsigned int tilePixelPosRow = threadIdx.y + subBlockNo * blockDim.y;
		unsigned int iPixelPosRow = tileStartRow + tilePixelPosRow;


		// Check if the pixel in the tile and image.
		// Note that the apron of the tile is excluded.
		if (iPixelPosCol >= tileStartCol + S && iPixelPosCol < tileEndClampedCol - S &&
			iPixelPosRow >= tileStartRow + S && iPixelPosRow < tileEndClampedRow - S) {

			// Compute the pixel position for the output image
			unsigned int oPixelPosCol = iPixelPosCol - S; // removing the origin
			unsigned int oPixelPosRow = iPixelPosRow - S;
			unsigned int oPixelPos = oPixelPosRow * W + oPixelPosCol;

			unsigned int tilePixelPos = tilePixelPosRow * tileW + tilePixelPosCol;

			d_h[oPixelPos] = 0.0;
			for (int i = -S; i <= S; i++) {
				for (int j = -S; j <= S; j++) {
					int tilePixelPosOffset = i * tileW + j;
					int coefPos = (i + S) * kernelSize + (j + S);
					d_h[oPixelPos] += sData[tilePixelPos + tilePixelPosOffset] * d_cFilterKernel[coefPos];
				}
			}

		}

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
void gaussianBlur_col_separable(unsigned char *d_in, unsigned char *d_out, 
        const int rows, const int cols, float *d_filter, const int filter_width){

	int pixVal;
	int current_col;
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;

	if (j < cols && i < rows){
		// Reset parameters
		pixVal = 0;

		// Obtain pixels right under the kernel
		for (int blur_c = -(filter_width / 2); blur_c <= (filter_width / 2); blur_c++) {

			// Calculate the index of the current row and that of the current column
			current_col = i + blur_c;

			// Boundary check
			if ((current_col >= 0) && (current_col < rows)) {
				pixVal += d_in[current_col*cols + j]*d_filter[(filter_width / 2) - blur_c];
			}
		}
	
		// Save the result
		//d_out[i*cols + j] = (unsigned char)(pixVal);
		d_out[i*cols + j] = d_in[i*cols + j];

	}	
	return;
} 

__global__ 
void gaussianBlur_row_separable(unsigned char *d_in, unsigned char *d_out, 
        const int rows, const int cols, float *d_filter, const int filter_width){

	int pixVal;
	int current_row;
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	if (j < cols && i < rows){

		// Reset parameters
		pixVal = 0;

		// Obtain pixels right under the kernel
		for (int blur_r = -(filter_width / 2); blur_r <= (filter_width / 2); blur_r++) {

		// Calculate the index of the current row and that of the current column
			current_row = j + blur_r;

			// Boundary check
			if ((current_row >= 0) && (current_row < cols)) {
				pixVal += d_in[i*cols + current_row]*d_filter[(filter_width / 2) - blur_r];
			}
		}

		//d_out[i * cols + j] = (unsigned char) pixVal;
		d_out[i*cols + j] = d_in[i*cols + j];
	}

	return;
} 

void gauss_blur_shared_mem(uchar4 *d_padded_image, int padded_width, int padded_height,
	float *d_filter, int filter_width, uchar4 *d_filterred_result, int img_width, int img_height,
	unsigned char *d_red, unsigned char *d_green, unsigned char *d_blue,
	unsigned char *d_rblurred, unsigned char *d_gblurred, unsigned char *d_bblurred)
{
	// Set the execution configuration
	const unsigned int blockW = 32;
	const unsigned int blockH = 32;
	const unsigned int tileW = blockW + 2 * 4;
	const unsigned int tileH = blockH + 2 * 4;
	const unsigned int threadBlockH = 8;

	const dim3 grid(int(ceil(img_width / blockW)+1), int(ceil(img_height / blockH)+1)); //int(ceil(cols / BLOCK))
	const dim3 threadBlock(tileW, threadBlockH);

	// Set the size of shared memory
	const unsigned int sharedMemorySizeByte = tileW * tileH * sizeof(float);

	// Separate the channels
	dim3 gridSize_padded(int(ceil(padded_width / blockW) + 1), int(ceil(padded_height / blockH) + 1), 1);
	dim3 blockSize_padded(int(blockW), int(blockH), 1);
	separateChannels <<<gridSize_padded, blockSize_padded >>>(d_padded_image, d_red, d_green, d_blue, int(padded_height), int(padded_width));
	cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());

	//
	imageFilteringKernel << <grid, threadBlock, sharedMemorySizeByte >> >(d_red, padded_width, padded_height,
		blockW, blockH, 4,
		d_rblurred, img_width, img_height, d_filter);
	cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());

	//
	imageFilteringKernel << <grid, threadBlock, sharedMemorySizeByte >> >(d_green, padded_width, padded_height,
		blockW, blockH, 4,
		d_gblurred, img_width, img_height, d_filter);
	cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());

	//
	imageFilteringKernel << <grid, threadBlock, sharedMemorySizeByte >> >(d_blue, padded_width, padded_height,
		blockW, blockH, 4,
		d_bblurred, img_width, img_height, d_filter);
	cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());

	// Recombine the channels
	dim3 gridSize_recombine(int(ceil(img_width / blockW) + 1), int(ceil(img_height / blockH) + 1), 1);
	dim3 blockSize_recombine(int(blockW), int(blockH), 1);
	recombineChannels << <gridSize_recombine, blockSize_recombine >> >(d_rblurred, d_gblurred, d_bblurred, d_filterred_result, int(img_height), int(img_width));
	cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());
}

void separable_gauss_blur(uchar4* d_imrgba, uchar4 *d_oimrgba, size_t rows, size_t cols, 
        unsigned char *d_red, unsigned char *d_green, unsigned char *d_blue, 
        unsigned char *d_rblurred, unsigned char *d_gblurred, unsigned char *d_bblurred,
        float *d_row_filter, float *d_col_filter,  int filterWidth){

        dim3 blockSize(BLOCK,BLOCK,1);
        dim3 gridSize((cols)/BLOCK,(rows)/BLOCK,1);

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
