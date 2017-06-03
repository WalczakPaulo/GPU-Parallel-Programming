#include "cuda_runtime.h"


__global__ void cuImageBrighten(const float *dev_image, 
				float *dev_out, int w, int h)
{
	int tx = threadIdx.x;   int ty = threadIdx.y;
	int bx = blockIdx.x;	int by = blockIdx.y;

	int pos = tx + 32*bx + w* ty + 32*w*by;
	dev_out[pos] = min(255.0f, dev_image[pos] + 50);
	__syncthreads();
}

__global__ void cuFilterImage(const float *image_in, float *image_out, int w, int h, int pad) {

	//dobray do paddingu
	const int BLOCK_SIZE = 32;
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bx = blockIdx.x;
	int by = blockIdx.y;
	const int w_padded = w + 2 * pad;
	const int h_padded = h + 2 * pad;
	const int offset = pad + pad * w_padded;
	const int pos = tx + 32 * bx + (ty + 32 * by) * w_padded + offset;
	const int pos2 = tx + bx * 32 + w * (ty + by * 32);

	__shared__ float pixels[3 * BLOCK_SIZE][3 * BLOCK_SIZE];

	for (int i = -1; i <= 1; i++) {
		for (int j = -1; j <= 1; j++) {
			int t = pos + i * BLOCK_SIZE + j * w_padded * BLOCK_SIZE;
			pixels[tx + (i + 1)*BLOCK_SIZE][ty + (j + 1)*BLOCK_SIZE] = image_in[t];
		}
	}
	__syncthreads();
	
	
	//maska
	const int g[] = { -1, -1, 0, 1, 1 };
	
	
	float xFilter = 0;
	float yFilter = 0;

	
	// jak na slajdach
#pragma UNROLL
	for (int i = -2; i <= 2; i++) {
		for (int j = -2; j <= 2; j++) {
			xFilter += pixels[BLOCK_SIZE + tx + i][BLOCK_SIZE + ty + j] * g[i + 2];
			yFilter += pixels[BLOCK_SIZE + tx + i][BLOCK_SIZE + ty + j] * g[j + 2];
		}
	}
	image_out[pos2] = max(0.0f, min(255.0f, sqrtf(xFilter*xFilter + yFilter*yFilter)));

	__syncthreads();
}



extern "C" bool cuImageProcessing(unsigned char *image, unsigned char *out_image, int w, int h)
{
	const int PADDING_SIZE = 32; // ramka, w sumie za duza
	// na orbazy
	float *pinned_input_image, *pinned_output_image;
	float *dev_input, *dev_output;

	const int w_padded = w + 2 * PADDING_SIZE;
	const int h_padded = h + 2 * PADDING_SIZE; // nowe rozmiary

	cudaHostAlloc<float>((float**)&pinned_input_image,
		w_padded*h_padded*sizeof(float), cudaHostAllocDefault);
	cudaHostAlloc<float>((float**)&pinned_output_image,
		w*h*sizeof(float), cudaHostAllocDefault);

		
	dim3 dimGrid(w/32, h/32);
	dim3 dimBlock(32, 32);	
		
	// kopiowanie do obrazow z paddngiem	
	for (int x = 0; x < w_padded; x++) {
		for (int y = 0; y < h_padded; y++) {
			if (x >= PADDING_SIZE && x < w + PADDING_SIZE && y >= PADDING_SIZE && y < h + PADDING_SIZE)
				pinned_input_image[x + y * w_padded] = image[(x - PADDING_SIZE) + (y - PADDING_SIZE)* w];
			else
				pinned_input_image[x + y * w_padded] = 0;
		}
	}


		

	cudaMalloc((void**)&dev_input, w_padded*h_padded* sizeof(float));
	cudaMalloc((void**)&dev_output, w*h * sizeof(float));
	cudaMemcpy(dev_input, pinned_input_image, w_padded*h_padded * sizeof(float), cudaMemcpyHostToDevice);

	

	// 10 razy, zeby sprawdzic czy sie nie wywala
	for (int i = 0; i < 10; i++)
	{
		cuFilterImage << <dimGrid, dimBlock >> >(dev_input, dev_output, w, h, PADDING_SIZE);
		cudaDeviceSynchronize();
	}

	cudaMemcpy(pinned_output_image, dev_output, w*h * sizeof(float), cudaMemcpyDeviceToHost);

	//kopia
	for (int i = 0; i<w*h; i++)
		out_image[i] = pinned_output_image[i];

	cudaFree(dev_input);
	cudaFree(dev_output);

	cudaFreeHost(pinned_input_image);
	cudaFreeHost(pinned_output_image);

	return true;
}
