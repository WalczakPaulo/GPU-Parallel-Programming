#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>


__global__ void saxpy(const int n, const float *const xt, const float a, float *const yt)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;  i += blockDim.x * gridDim.x) 
		{
		yt[i] += a*xt[i];
		}
}



int main() try {
	const long long GiB = 1012 * 1024 * 1024;
	const float memvavail = 1; //! memory
	const float memvalock = 0.8*memvavail;
	const long long N= memvalock*GiB / sizeof(float) / 2;
	
	std::vector<float> vx(N);
	std::vector<float> vy(N);
	float *dev_x = 0;
    float *dev_y = 0;
	cudaError_t cudaStatus;
	cudaEvent_t startEvent, stopEvent;

	//std::vector<float> vz(N);
	//std::fill(begin(vx), end(vx), -1.0f);
	//std::fill(begin(vy), end(vy),  1.0f);
//	const long long memvalock = N * 2.0 * sizeof(float) / GiB;
	std::cout << "  memory requirement [GiB] " << memvalock << std::endl;
	//const long long memvavail = 32; //! memory
	if (0.81*memvavail < memvalock)
		throw std::runtime_error("Not enough memory to proceed!");
	//std::fill(begin(vz), end(vz), 1.0f);
	const float a = 1.0f;
	const float b = -1.0f;
	//std::vector<std::thread> vt;
	std::vector<float> vtime;
	const int MAXITER = 10;
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
		throw std::runtime_error("cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
    }
	cudaStatus =  cudaEventCreate(&startEvent);
    if (cudaStatus != cudaSuccess) {
		throw std::runtime_error("cudaEventCreate failed!");
    }
	cudaStatus =  cudaEventCreate(&stopEvent);
    if (cudaStatus != cudaSuccess) {
		throw std::runtime_error("cudaEventCreate failed!");
    }
    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_x, N * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        throw std::runtime_error("cudaMalloc failed!");
    }

    cudaStatus = cudaMalloc((void**)&dev_y, N * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        throw std::runtime_error("cudaMalloc failed!");

    }
		std::fill(begin(vx), end(vx), -1.0f);

	for (int i = 0; i < MAXITER; ++i) {
		std::fill(begin(vy), end(vy), 1.0f);
    // Choose which GPU to run on, change this on a multi-GPU system.


    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_x, &vx.front(), N * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        throw std::runtime_error("cudaMemcpy failed!");

    }

    cudaStatus = cudaMemcpy(dev_y, &vy.front(), N * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        throw std::runtime_error("cudaMemcpy failed!");
    }

    // Launch a kernel on the GPU with one thread for each element.
	cudaEventRecord(startEvent,0);
    if (cudaStatus != cudaSuccess) {
        throw std::runtime_error("cudaEventRecord failed!");
    }
	const int grid_size = 2048;
	const int threads_nb = 256;
	//(N+255)/256
    saxpy<<<grid_size, threads_nb>>>(N, dev_x, a, dev_y);
	cudaStatus = cudaGetLastError();
	if(cudaStatus != cudaSuccess) {
       throw std::runtime_error(cudaGetErrorString(cudaStatus));
	 }
	cudaStatus = cudaEventRecord(stopEvent,0);
    if (cudaStatus != cudaSuccess) {
        throw std::runtime_error("cudaEventRecord failed!");
    }
	cudaStatus = cudaEventSynchronize(stopEvent);
    if (cudaStatus != cudaSuccess) {
        throw std::runtime_error("cudaEventSynchronize failed!");
    }
	float durms=0;
	cudaStatus = cudaEventElapsedTime(&durms, startEvent, stopEvent);
    if (cudaStatus != cudaSuccess) {
        throw std::runtime_error("cudaEventElapsedTime failed!");
    }
	cudaStatus = cudaMemcpy(&vy.front(),dev_y, N * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        throw std::runtime_error("cudaMemcpy failed!");
    }
	const float s = std::accumulate(begin(vy), end(vy), 0.0f); // to check - should be 0
	std::cout <<  "  duration [ms] = " << durms << "  sum = " << s << std::endl;
	vtime.push_back(durms);


	}
	const double tmean = static_cast<double>(std::accumulate(begin(vtime), end(vtime), 0.0))/vtime.size();
	const double tsdev = std::sqrt(std::inner_product(begin(vtime), end(vtime), begin(vtime), 0.0) / vtime.size() - tmean*tmean);
	std::cout << "  duration mean [ms] = " << tmean << "  sdev = " << tsdev << std::endl;
	const int io_op = 3; //number of io operation
	const int data_type_size = sizeof(float); //data size
	const double bandwidth = static_cast<double>(N * io_op* data_type_size) / GiB / tmean * 1000;
	const int fpoper = 2; //number of fp operation
	const double ai = static_cast<double>(fpoper) / (io_op *data_type_size); //arithmetic intensity
	const double compower = ai*bandwidth;
	std::cout << "  bandwidth [GiB/s]= " << bandwidth << std::endl;
	std::cout << "  compower [GiFlops]= " << compower << std::endl;
	const double effective_mem_freq = 4 * 1.001;
  	const double interface_width = 256;
	const double bandwidth_theory = effective_mem_freq * interface_width / 8;
	const double cpuclock = 0.81;
	const int core_nb = 336;
	const int instr_per_cycle = 2;
	const double compower_theory = cpuclock *core_nb* instr_per_cycle;
	std::cout << "  bandwidth theory [GiB/s]= " << bandwidth_theory << std::endl;
	std::cout << "  compower theory [GiFlops]= " << compower_theory << std::endl;
	const double bandwidth_eff = bandwidth / bandwidth_theory * 100; //bandwidth efficiency in %
	const double compower_eff = compower / compower_theory * 100; //compower efficiency in %
	std::cout << "  bandwidth efficiency [%]= " << bandwidth_eff << std::endl;
	std::cout << "  compower efficiency [%]= " << compower_eff << std::endl;
	std::cin.get();

}
catch(std::exception &e) {
	std::cerr << e.what() <<"\n";
}
catch(...) {
	std::cerr << "unhandled exception\n";
}
