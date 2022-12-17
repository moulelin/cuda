#include <stdio.h>
__global__ void hello_from_gpu(){
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    printf("this is from gpu and the block is %d, the thread is %d\n", bid, tid);
}
int main(void){
    hello_from_gpu<<<2,3>>>();
    cudaDeviceSynchronize();
    return 0;
}