#include <stdio.h>
void __global__ add ( double *d_x,  double *d_y,  double *d_z);
double __device__ add_device(const double x, double y); //device修饰可以有返回值
double a = 1.3;
double b = 1.4;
double c = 2.7;
//global是host调用，可以看作是device的入口。
int main(){

    int N = 10000000;
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
    int block_size = 128;
    int grid_size = (N-1)/block_size + 1; // 为了当除不整的时候，多分配一个块，比如11个数，每个块大小5，就分三个
    add<<<grid_size, block_size>>>(d_x, d_y, d_z);//也可以直接传入N或者size
    cudaMemcpy(h_z, d_z, size, cudaMemcpyDeviceToHost);
  //  check(h_z, N);
    double sum = 0.0;
    for (int i=0;i<N;i++){
        sum += h_z[i];
    }
    cudaDeviceSynchronize();
    printf("the sum is %f,%f", sum,h_z[0]);
    free(h_x);
    free(h_y);
    free(h_z);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);
    return 0;
}

void __global__ add( double *d_x,  double *d_y,  double *d_z){

    const int n = blockDim.x*blockIdx.x+threadIdx.x;
    d_z[n] = add_device(d_x[n] , d_y[n]);
}
double __device__ add_device(const double x, const double y){
    return x+y;
}