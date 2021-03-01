#include <cuda_runtime.h> 
#include <cuda.h> 
#include "utils.h"

/*
 * The launcher for your kernels. 
 * This is a single entry point and 
 * all arrays **MUST** be pre-allocated 
 * on device. you must implement all other 
 * kernels in the respective files.
 * */ 



//void your_gauss_blur(uchar4* d_imrgba, uchar4 *d_oimrgba, size_t rows, size_t cols, 
//        unsigned char *d_red, unsigned char *d_green, unsigned char *d_blue, 
//        unsigned char *d_rblurred, unsigned char *d_gblurred, unsigned char *d_bblurred,  
//        float *d_filter, int filterWidth);

// The function using GPU shared memory
void gauss_blur_shared_mem(uchar4 *d_padded_image, int padded_width, int padded_height,
	float *d_filter, int filter_width, uchar4 *d_filterred_result, int img_width, int img_height,
	unsigned char *d_red, unsigned char *d_green, unsigned char *d_blue,
	unsigned char *d_rblurred, unsigned char *d_gblurred, unsigned char *d_bblurred);

void separable_gauss_blur(uchar4* d_imrgba, uchar4 *d_oimrgba, size_t rows, size_t cols, 
        unsigned char *d_red, unsigned char *d_green, unsigned char *d_blue, 
        unsigned char *d_rblurred, unsigned char *d_gblurred, unsigned char *d_bblurred,
        float *d_row_filter, float *d_col_filter,  int filterWidth);
