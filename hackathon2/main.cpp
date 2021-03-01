#include <iostream>
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h> 
#include <cassert>
#include <cstdio> 
#include <string> 
#include <opencv2/opencv.hpp> 
#include <cmath> 
#include <chrono>

#include "utils.h"
#include "gaussian_kernel.h"


/* 
 * Compute if the two images are correctly 
 * computed. The reference image can 
 * either be produced by a software or by 
 * your own serial implementation.
 * */
void checkApproxResults(unsigned char *ref, unsigned char *gpu, size_t numElems){

    for(size_t i = 0; i < numElems; i++){
        if(ref[i] - gpu[i] > 1e-5){
            std::cerr << "Error at position " << i << "\n"; 

            std::cerr << "Reference:: " << std::setprecision(17) << +ref[i] <<"\n";
            std::cerr << "GPU:: " << +gpu[i] << "\n";

            exit(1);
        }
    }
}



void checkResult(const std::string &reference_file, const std::string &output_file, float eps){
    cv::Mat ref_img, out_img; 

    ref_img = cv::imread(reference_file, -1);
    out_img = cv::imread(output_file, -1);


    unsigned char *refPtr = ref_img.ptr<unsigned char>(0);
    unsigned char *oPtr = out_img.ptr<unsigned char>(0);

    checkApproxResults(refPtr, oPtr, ref_img.rows*ref_img.cols*ref_img.channels());
    std::cout << "PASSED!\n";


}

// Pad the image apron
template <typename T>
int replicationPadding(T *image, const unsigned int &iWidth, const unsigned int &iHeight,
	const unsigned int &hFilterSize,
	T *paddedImage, const unsigned int &paddedIWidth, const unsigned int &paddedIHeight)
{

	//
	// Perform extended padding
	//
	for (unsigned int i = 0; i < paddedIHeight; i++) {
		for (unsigned int j = 0; j < paddedIWidth; j++) {

			// Set the pixel position of the extended image
			unsigned int extendedPixelPos = i * paddedIWidth + j;

			// Set the pixel position of the input image
			unsigned int pixelPos = 0;
			if (j >= 0 && j < hFilterSize &&
				i >= 0 && i < hFilterSize) { // (The top left corner)
				pixelPos = 0;
			}
			else if (j >= hFilterSize && j < iWidth + hFilterSize &&
				i >= 0 && i < hFilterSize) { // (The top padded area)
				pixelPos = j - hFilterSize;
			}
			else if (j >= iWidth + hFilterSize && j < iWidth + 2 * hFilterSize &&
				i >= 0 && i < hFilterSize) { // (The top right corner)
				pixelPos = iWidth - 1;
			}
			else if (j >= 0 && j < hFilterSize &&
				i >= hFilterSize && i < iHeight + hFilterSize) { // (The left padded area)
				pixelPos = (i - hFilterSize) * iWidth;
			}
			else if (j >= hFilterSize && j < iWidth + hFilterSize &&
				i >= hFilterSize && i < iHeight + hFilterSize) { // (The original image)
				pixelPos = (i - hFilterSize) * iWidth + (j - hFilterSize);
			}
			else if (j >= iWidth + hFilterSize && j < iWidth + 2 * hFilterSize &&
				i >= hFilterSize && i < iHeight + hFilterSize) { // (The right padded area)
				pixelPos = (i - hFilterSize) * iWidth + (iWidth - 1);
			}
			else if (j >= 0 && j < hFilterSize &&
				i >= iHeight + hFilterSize && i < iHeight + 2 * hFilterSize) { // (The bottom left corner)
				pixelPos = (iHeight - 1) * iWidth;
			}
			else if (j >= hFilterSize && j < iWidth + hFilterSize &&
				i >= iHeight + hFilterSize && i < iHeight + 2 * hFilterSize) { // (The bottom padded area)
				pixelPos = (iHeight - 1) * iWidth + (j - hFilterSize);
			}
			else if (j >= iWidth + hFilterSize && j < iWidth + 2 * hFilterSize &&
				i >= iHeight + hFilterSize && i < iHeight + 2 * hFilterSize) { // (The bottom right corner)
				pixelPos = (iHeight - 1) * iWidth + (iWidth - 1);
			}

			// Copy the pixel value
			paddedImage[extendedPixelPos] = image[pixelPos];

		}
	}

	return 0;

}

// f_sz is the dimension of the kernel
void gaussian_blur_filter(float *arr, const int f_sz, const float f_sigma=0.2){ 
    float filterSum = 0.f;
    float norm_const = 0.0; // normalization const for the kernel 

    for(int r = -f_sz/2; r <= f_sz/2; r++){
        for(int c = -f_sz/2; c <= f_sz/2; c++){
            float fSum = expf(-(float)(r*r + c*c)/(2*f_sigma*f_sigma)); 
            arr[(r+f_sz/2)*f_sz + (c + f_sz/2)] = fSum; 
            filterSum  += fSum;
        }
    } 

    norm_const = 1.f/filterSum; 

    for(int r = -f_sz/2; r <= f_sz/2; ++r){
        for(int c = -f_sz/2; c <= f_sz/2; ++c){
            arr[(r+f_sz/2)*f_sz + (c + f_sz/2)] *= norm_const;
        }
    }
}

// f_sz is the dimension of the kernel
void gaussian_blur_row_filter(float *arr, const int f_sz, const float f_sigma=0.2){ 
    float filterSum = 0.f;
    float norm_const = 0.0; // normalization const for the kernel 

    for(int r = -f_sz/2; r <= f_sz/2; r++){
            float fSum = expf(-(float)(r*r)/(2*f_sigma*f_sigma)); 
            arr[(r+f_sz/2)] = fSum; 
            filterSum  += fSum;
    } 

    norm_const = 1.f/filterSum; 

    for(int r = -f_sz/2; r <= f_sz/2; ++r){
            arr[(r+f_sz/2)] *= norm_const;
    }
}

// f_sz is the dimension of the kernel
void gaussian_blur_col_filter(float *arr, const int f_sz, const float f_sigma=0.2){ 
    float filterSum = 0.f;
    float norm_const = 0.0; // normalization const for the kernel 

    for(int c = -f_sz/2; c <= f_sz/2; c++){
            float fSum = expf(-(float)(c*c)/(2*f_sigma*f_sigma)); 
            arr[(c+f_sz/2)] = fSum; 
            filterSum  += fSum;
    } 

    norm_const = 1.f/filterSum; 

    for(int c = -f_sz/2; c <= f_sz/2; ++c){
            arr[(c+f_sz/2)] *= norm_const;
    }
}

// Serial implementations of kernel functions
void serialGaussianBlur(unsigned char *in, unsigned char *out, const int rows, const int cols, 
    float *filter, const int filterWidth){

	int current_row = 0;
	int current_col = 0;
	float pixVal = 0;

	// Go through the whole image
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {

			// Reset parameters
			pixVal = 0;

			// Obtain pixels right under the kernel
			for (int blur_r = -(filterWidth / 2); blur_r <= (filterWidth / 2); blur_r++) {
				for (int blur_c = -(filterWidth / 2); blur_c <= (filterWidth / 2); blur_c++) {

					// Calculate the index of the current row and that of the current column
					current_row = i + blur_r;
					current_col = j + blur_c;

					// Boundary check
					if ((current_row >= 0) && (current_row < rows) && (current_col >= 0) && (current_col < cols)) {
						pixVal += in[current_row*cols + current_col]*filter[(blur_r + filterWidth / 2)*filterWidth + (blur_c + filterWidth / 2)];
					}
				}
			}

			// Save the result
			out[i*cols + j] = (unsigned char)(pixVal);
		}
	}
} 

// Serial implementations of kernel functions
void serialGaussianBlur_row(unsigned char *in, unsigned char *out, const int rows, const int cols, 
    float *filter, const int filterWidth){

	int current_row = 0;
	float pixVal = 0;

	// Go through the whole image
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {

			// Reset parameters
			pixVal = 0;

			// Obtain pixels right under the kernel
			for (int blur_r = -(filterWidth / 2); blur_r <= (filterWidth / 2); blur_r++) {

				// Calculate the index of the current row and that of the current column
				current_row = j + blur_r;

				// Boundary check
				if ((current_row >= 0) && (current_row < cols)) {
					pixVal += in[i*cols + current_row]*filter[(filterWidth / 2) - blur_r];
				}
			}

			// Save the result
			out[i*cols + j] = (unsigned char)(pixVal);
		}
	}
} 

// Serial implementations of kernel functions
void serialGaussianBlur_col(unsigned char *in, unsigned char *out, const int rows, const int cols, 
    float *filter, const int filterWidth){

	int current_col = 0;
	float pixVal = 0;

	// Go through the whole image
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {

			// Reset parameters
			pixVal = 0;

			// Obtain pixels right under the kernel
			for (int blur_c = -(filterWidth / 2); blur_c <= (filterWidth / 2); blur_c++) {

				// Calculate the index of the current row and that of the current column
				current_col = i + blur_c;

				// Boundary check
				if ((current_col >= 0) && (current_col < rows)) {
					pixVal += in[current_col*cols + j]*filter[(filterWidth / 2) - blur_c];
				}
			}
		
			// Save the result
			out[i*cols + j] = (unsigned char)(pixVal);
		}
	}
}

void serialSeparateChannels(uchar4 *imrgba, unsigned char *r, unsigned char *g, unsigned char *b,
    const int rows, const int cols){

	// Separate the image channels
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			r[i*cols + j] = imrgba[i*cols + j].x;
			g[i*cols + j] = imrgba[i*cols + j].y;
			b[i*cols + j] = imrgba[i*cols + j].z;
		}
	}

} 

void serialRecombineChannels(unsigned char *r, unsigned char *g, unsigned char *b, uchar4 *orgba,
    const int rows, const int cols){

	// Combine the 3 channels
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			orgba[i*cols + j].x = r[i*cols + j];
			orgba[i*cols + j].y = g[i*cols + j];
			orgba[i*cols + j].z = b[i*cols + j];
			orgba[i*cols + j].w = 255;
		}
	}
} 


int main(int argc, char const *argv[]) {
   
    uchar4 *h_in_img, *h_o_img, *r_o_img; // pointers to the actual image input and output pointers  
    uchar4 *d_in_img, *d_o_img;
    uchar4 *d_sep_in_img;

    unsigned char *h_red, *h_blue, *h_green; 
	unsigned char *h_red_blurred, *h_green_blurred, *h_blue_blurred;
    unsigned char *d_red, *d_blue, *d_green;   
    unsigned char *d_red_blurred, *d_green_blurred, *d_blue_blurred;   
    unsigned char *d_sep_red, *d_sep_blue, *d_sep_green;   

    float *h_filter, *d_filter;  
    float *h_col_filter, *d_col_filter;  
    float *h_row_filter, *d_row_filter;  
    cv::Mat imrgba, o_img, output_gpu_final, reference_image; 

    const int fWidth = 9; 
    double fDev = 2;
	int64_t time_diff;
	float time_diff_float;
	std::chrono::high_resolution_clock::time_point start_time, end_time;
	std::string image_name = "monarch";
    std::string infile = "./" + image_name + ".png";
    std::string outfile = "./"+ image_name + "(GPU).png";
    std::string reference = "./"+ image_name + "(serial).png";

    // Read the original image
    cv::Mat img = cv::imread(infile.c_str(), cv::IMREAD_COLOR); 
    if(img.empty()){
        std::cerr << "Image file couldn't be read, exiting\n"; 
        return 1;
    }

	// Convert BGR image to RGBA image
    cv::cvtColor(img, imrgba, cv::COLOR_BGR2RGBA);

	// The pointer to input image
	h_in_img = imrgba.ptr<uchar4>(0);

	// filter allocation 
	h_filter = new float[fWidth*fWidth];
	h_row_filter = new float[fWidth];
	h_col_filter = new float[fWidth];
	gaussian_blur_filter(h_filter, fWidth, float(fDev)); // create a filter of 9x9 with std_dev = 2  
	gaussian_blur_row_filter(h_row_filter, fWidth, float(fDev)); // create a filter of 9x9 with std_dev = 2  
	gaussian_blur_col_filter(h_col_filter, fWidth, float(fDev)); // create a filter of 9x9 with std_dev = 2  

	// Print the parameters within the filter
	printArray<float>(h_filter, 81); // printUtility.
	printArray<float>(h_filter, 9); // printUtility.
	printArray<float>(h_filter, 9); // printUtility.

	// Create an apron for the input image
	unsigned int paddedIWidth = img.cols + 2 * 4;
	unsigned int paddedIHeight = img.rows + 2 * 4;
	uchar4* h_paddedImage;
	h_paddedImage = new uchar4[paddedIWidth * paddedIHeight]; // Allocate memory
	replicationPadding(h_in_img, img.cols, img.rows,4,h_paddedImage, paddedIWidth, paddedIHeight);

	//// test
	//cv::Mat test_single_2;
	//cv::Mat test_single_1(paddedIHeight, paddedIWidth, CV_8UC4, h_paddedImage);
	//cv::cvtColor(test_single_1, test_single_2, cv::COLOR_RGBA2BGR);
	//cv::imshow("Padded", test_single_2);
	//cv::waitKey(0);

	// (GPU) The number of pixels within the image
	const size_t  numPixels = img.rows*img.cols;
	const size_t numPaddedPixels = paddedIHeight * paddedIWidth;

	// (GPU) The number of parameters within the kernel
	const size_t numFilterParam = fWidth*fWidth;

	// (GPU) Allocate memory on GPU
	checkCudaErrors(cudaMalloc((void**)&d_in_img, sizeof(uchar4)*numPaddedPixels));
	checkCudaErrors(cudaMalloc((void**)&d_red, sizeof(unsigned char)*numPaddedPixels));
	checkCudaErrors(cudaMalloc((void**)&d_green, sizeof(unsigned char)*numPaddedPixels));
	checkCudaErrors(cudaMalloc((void**)&d_blue, sizeof(unsigned char)*numPaddedPixels));
	checkCudaErrors(cudaMalloc((void**)&d_red_blurred, sizeof(unsigned char)*numPixels));
	checkCudaErrors(cudaMalloc((void**)&d_green_blurred, sizeof(unsigned char)*numPixels));
	checkCudaErrors(cudaMalloc((void**)&d_blue_blurred, sizeof(unsigned char)*numPixels));
	checkCudaErrors(cudaMalloc((void**)&d_o_img, sizeof(uchar4)*numPixels));
	checkCudaErrors(cudaMalloc((void**)&d_filter, sizeof(float)*numFilterParam));

	checkCudaErrors(cudaMalloc((void**)&d_sep_in_img, sizeof(uchar4)*numPixels));
	checkCudaErrors(cudaMalloc((void**)&d_sep_red, sizeof(unsigned char)*numPixels));
	checkCudaErrors(cudaMalloc((void**)&d_sep_green, sizeof(unsigned char)*numPixels));
	checkCudaErrors(cudaMalloc((void**)&d_sep_blue, sizeof(unsigned char)*numPixels));
	checkCudaErrors(cudaMalloc((void**)&d_row_filter, sizeof(float)*fWidth));
	checkCudaErrors(cudaMalloc((void**)&d_col_filter, sizeof(float)*fWidth));

	// (GPU) Create a matrix to store the output image from GPU
	o_img.create(img.rows, img.cols, CV_8UC4);
	h_o_img = o_img.ptr<uchar4>(0);

	// (GPU) Copy the data from host to device
	checkCudaErrors(cudaMemcpy(d_in_img, h_paddedImage, sizeof(uchar4)*numPaddedPixels, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_sep_in_img, h_in_img, sizeof(uchar4)*numPixels, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_filter, h_filter, sizeof(float)*numFilterParam, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_row_filter, h_row_filter, sizeof(float)*fWidth, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_col_filter, h_col_filter, sizeof(float)*fWidth, cudaMemcpyHostToDevice));

	// (GPU) shared kernel launch code 
	gauss_blur_shared_mem(d_in_img,paddedIWidth,paddedIHeight,d_filter, fWidth,d_o_img,(img.cols),(img.rows),
		d_red,d_green,d_blue,d_red_blurred,d_green_blurred,d_blue_blurred);

	  //original kernel
		original_gauss_blur(d_sep_in_img,d_o_img,img.rows,img.cols,d_sep_red,d_sep_green,d_sep_blue \
			,d_red_blurred,d_green_blurred,d_blue_blurred, d_filter, fWidth);

	// (GPU) separable kernel launch 
	separable_gauss_blur(d_sep_in_img,d_o_img,img.rows,img.cols,d_sep_red,d_sep_green,d_sep_blue \
			,d_red_blurred,d_green_blurred,d_blue_blurred, d_row_filter, d_col_filter, fWidth);

	//// (GPU) Copy the data from device to host
	checkCudaErrors(cudaMemcpy(h_o_img, d_o_img, numPixels * sizeof(uchar4), cudaMemcpyDeviceToHost));

	//// (GPU) Create and save the image with the output data 
	bool suc_gpu = false;
	cv::Mat output_gpu(img.rows, img.cols, CV_8UC4, (void*)h_o_img);
	cv::cvtColor(output_gpu, output_gpu_final, cv::COLOR_RGBA2BGR);
	suc_gpu = cv::imwrite(outfile.c_str(), output_gpu_final);
	if(!suc_gpu){
	      std::cerr << "Couldn't write GPU image!\n";
	       return 1;
	}


	// test
	//cv::imshow("Output from GPU", output_gpu_final);
	//cv::waitKey(0);

	// (Serial) Record the start time. 
	start_time = std::chrono::high_resolution_clock::now();

	// (Serial) allocate memory
	h_red = new unsigned char[(img.rows)*(img.cols)];
	h_green = new unsigned char[(img.rows)*(img.cols)];
	h_blue = new unsigned char[(img.rows)*(img.cols)];
	h_red_blurred = new unsigned char[(img.rows)*(img.cols)];
	h_green_blurred = new unsigned char[(img.rows)*(img.cols)];
	h_blue_blurred = new unsigned char[(img.rows)*(img.cols)];
	r_o_img = new uchar4[(img.rows)*(img.cols)];

	// (Serial) Separate the image to channels
	serialSeparateChannels(h_in_img, h_red, h_green, h_blue, img.rows, img.cols);

	//// test
	//// (GPU_1) Upload the padded image to GPU
	//unsigned char *d_paddedImage;
	//unsigned int paddedImageSizeByte = paddedIWidth * paddedIHeight * sizeof(unsigned char);
	//checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_paddedImage), paddedImageSizeByte));
	//checkCudaErrors(cudaMemcpy(d_paddedImage, h_paddedImage, paddedImageSizeByte, cudaMemcpyHostToDevice));

	//// test
	//// (GPU_1) call the kernel function for image filtering
	//unsigned char *d_filteringResult;
	//const unsigned int imageSizeByte = (img.rows) * (img.cols) * sizeof(unsigned char);
	//checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_filteringResult), imageSizeByte));
	//gauss_blur_shared_mem(d_paddedImage, paddedIWidth,paddedIHeight,d_filter, fWidth,d_filteringResult,img.cols, img.rows);
	//cv::Mat test_o_img;
	//unsigned char *test_h_o_img;
	//test_o_img.create(img.rows, img.cols, CV_8UC1);
	//test_h_o_img = test_o_img.ptr<unsigned char>(0);
	//checkCudaErrors(cudaMemcpy(test_h_o_img, d_filteringResult, numPixels * sizeof(unsigned char), cudaMemcpyDeviceToHost));
	//cv::Mat test_single(img.rows, img.cols, CV_8UC1, test_h_o_img);
	//cv::imshow("After filtering", test_single);
	//cv::waitKey(0);

	// (Serial) Apply Gaussian blur to the r channel
	serialGaussianBlur_row(h_red, h_red_blurred, img.rows, img.cols, h_row_filter, fWidth);
	serialGaussianBlur_col(h_red_blurred, h_red, img.rows, img.cols, h_col_filter, fWidth);

	// (Serial) Apply Gaussian blur to the g channel
	serialGaussianBlur_row(h_green, h_green_blurred, img.rows, img.cols, h_row_filter, fWidth);
	serialGaussianBlur_col(h_green_blurred, h_green, img.rows, img.cols, h_col_filter, fWidth);

	// (Serial) Apply Gaussian blur to the b channel
	serialGaussianBlur_row(h_blue, h_blue_blurred, img.rows, img.cols, h_row_filter, fWidth);
	serialGaussianBlur_col(h_blue_blurred, h_blue, img.rows, img.cols, h_col_filter, fWidth);

	//// test
	//cv::Mat test_single(img.rows, img.cols, CV_8UC1, h_red_blurred); 
	//cv::Mat test_single_1(img.rows, img.cols, CV_8UC1, h_red);
	////double minVal;
	////double maxVal;
	////cv::Point minLoc;
	////cv::Point maxLoc;
	////minMaxLoc(test_single, &minVal, &maxVal, &minLoc, &maxLoc);
	////std::cout << "min val: " << minVal << std::endl;
	////std::cout << "max val: " << maxVal << std::endl;
	//cv::imshow("Original", test_single_1);
	//cv::imshow("Modified", test_single);
	//cv::waitKey(0);

	// (Serial) Combine the channels to a image
	serialRecombineChannels(h_red, h_green, h_blue, r_o_img, img.rows, img.cols);

	// (Serial) Record the end time. 
	end_time = std::chrono::high_resolution_clock::now();

	// Print the elapsed time
	time_diff = std::chrono::duration_cast<std::chrono::microseconds> (end_time - start_time).count();
	time_diff_float = float(time_diff) / 1000000;
	std::cout << "Time for serial implementaion: " << time_diff_float << std::endl;

	// (Serial) Save the serial output image
	bool suc_serial = false;
	cv::Mat output_s(img.rows, img.cols, CV_8UC4, r_o_img);
	cv::cvtColor(output_s, reference_image, cv::COLOR_RGBA2BGR);
	suc_serial = cv::imwrite(reference.c_str(), reference_image);
	if(!suc_serial){
	       std::cerr << "Couldn't write serial image!\n";
	       return 1;
	}

	//// test
	//imshow("The serial implementation result", reference_image);
	//cv::waitKey(0);

 //   // check if the caclulation was correct to a degree of tolerance
 //   checkResult(reference, outfile, 1e-5);

	// (Serial) free any necessary memory.
	delete[] h_filter;
	delete[] h_red;
	delete[] h_green;
	delete[] h_blue;
	delete[] h_red_blurred;
	delete[] h_green_blurred;
	delete[] h_blue_blurred;
	delete[] r_o_img;

	//test
	delete[] h_paddedImage;

	// (GPU) free any necessary memory.
	cudaFree(d_in_img);
	cudaFree(d_o_img);
	cudaFree(d_red);
	cudaFree(d_green);
	cudaFree(d_blue);
	cudaFree(d_red_blurred);
	cudaFree(d_green_blurred);
	cudaFree(d_blue_blurred);
	cudaFree(d_filter);

	//test
	//cudaFree(d_paddedImage);
	//cudaFree(d_filteringResult);

//	system("pause");
    return 0;
}



