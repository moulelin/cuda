#include <stdio.h>
void __global__ add (const double *d_x, const double *d_y, const double *d_z);
void __device__ check(const double x, double y);
int a = 1.3;
int b = 1.4;
int c = 2.7;
//global是host调用，可以看作是device的入口。
int main(){

    int N = 1000000;
    double size = sizeof(double)*N;
    double *h_x = (double *)malloc(size);
    double *h_y = (double *)malloc(size);
    double *h_z = (double *)malloc(size);



    for (int i=0;i<N;i++){
        h_x[i] = a;
        h_y[i] = b;
    }
    double *d_x, *d_y, *d_z; 
    cudaMalloc((void **)&d_x, size);
    cudaMalloc((void **)&d_y, size);
    cudaMalloc((void **)&d_z, size);

    cudaMemcpy(d_x, h_x, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, size, cudaMemcpyHostToDevice);
    block_size = 128;
    grid_size = (N-1)/block_size +1;
    add<<<grid_size, block_size>>>(d_x, d_y, d_z);
    cudaMemcpy(h_z, d_z, size, cudaMemcpyDeviceToHost);
    check(h_z, N);
    double sum = 0;
    for (int i=0;i<N;i++){
        sum += h_z[i];
    }
    printf("the sum is %f", sum);
    return 0;
}

void __global__ add(const double *d_x, const double *d_y, const double *d_z){

    const int n = blockDim.x*blockIdx.x+threadIdx.x;
    d_z[n] = add_device(d_x[n] + d_y[n]);
}
double __device__ add_device(const double x, const double y){
    return x+y;
}