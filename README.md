# Serial and CUDA Implementation of Gaussian Blur

- This project applies a gaussian blur filter to an input image.
- It performs both serial and CUDA versions of this operation and compares the outputs for correctness.
- This project uses gcc 8.3.1, CUDA 11.0.2, and OpenCV 4.2.0

## Compilation (on Clemson's Palmetto cluster):
### Basic Compilation:
`> source envvars.sh # add required modules`
`> make              # create executable`

### Compilation for PBS Scripts:
`> source envvars.sh # add required modules`
`> sh compileall.sh  # compile executables for K20, P100 and V100 GPUs`

## To Run:
### Basic Run:
`> ./gblur <inputimage> <gpu_outputimage> <serial_outputimage>`

### Profiling Run:
`> qsub project2_<gpu_type>.pbs`
