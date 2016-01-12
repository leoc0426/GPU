#include "gpumain.h"
float *dens;          // host density
float *xv;            // host velocity in x
float *yv;            // host velocity in y
float *zv;            // host velocity in z
float *press;         // host pressure
float *temperature;   // host temperature

float *d_dens;        // device density
float *d_xv;          // device velocity in x
float *d_yv;          // device velocity in y
float *d_zv;          // device velocity in z
float *d_press;       // device pressure

float *U;

float *d_U;
float *d_E;
float *d_F;
float *d_G;
float *d_U_new;

float *d_FL;
float *d_FR;
float *d_FB;
float *d_FF;
float *d_FD;
float *d_FU;

float *h_body;
float *d_body;

__global__ void CalculateFlux(float* body, float* dens, float* xv, float* yv, float* zv, float* press,
                              float* E, float* F, float* G, 
                              float* FL, float* FR, float* FB, float* FF, float* FD, float* FU, 
                              float* U, float* U_new);

void Load_Dat_To_Array(char* input_file_name, float* body) {
	FILE *pFile;
	pFile = fopen(input_file_name, "r");
	if (!pFile) { printf("Open failure"); }

	int x, y, z;
	for (z = 0; z < NZ; z++) {
		for (y = 0; y < NY; y++) {
			for (x = 0; x < NX; x++) {
				h_body[z * NX * NY + y * NX + x] = -1.0;
			}
		}
	}
	float tmp1, tmp2, tmp3;
	int idx = 0;
	char line[60];

	// According to the 50x50x50 order
	for (x = 75; x > 25; x--) {
		for (z = 25; z < 75; z++) {
			for (y = 25; y < 75; y++) {
				idx = z * NX * NY + y * NX + x;
				fscanf(pFile, "%f%f%f%f", &tmp1, &tmp2, &tmp3, &body[idx]);
				/* test... 0.040018	 -0.204846	 -0.286759	 -1 */
				//if(body[idx] == 0.0) { 
				//	printf("%d, %f\n", idx, body[idx]);
				//	system("pause"); 
				//}
			}
		}
	}
	fclose(pFile);
}

void Init() {
	int i, j, k, z;
	for (i = 0; i < NX; i++) {
		for (j = 0; j < NY; j++) {
			for (k = 0; k < NZ; k++) {
				if (h_body[i + j*NX + k*NX*NY] == 0.0) { // body
					dens[i + j*NX + k*NX*NY] = 1.0;
					xv[i + j*NX + k*NX*NY] = 1.0;
					yv[i + j*NX + k*NX*NY] = 1.0;
					zv[i + j*NX + k*NX*NY] = 1.0;;
					press[i + j*NX + k*NX*NY] = 1.0;
					temperature[i + j*NX + k*NX*NY] = 1.0;
				}
				else {
					// air reference: http://www.engineeringtoolbox.com/standard-atmosphere-d_604.html
					dens[i + j*NX + k*NX*NY] = 0.00082 * 0.1;				// unit: kg / m^3
					xv[i + j*NX + k*NX*NY] = -7000;						// unit: m / s
					yv[i + j*NX + k*NX*NY] = 0;						// unit: m / s
					zv[i + j*NX + k*NX*NY] = 0;						// unit: m / s
					press[i + j*NX + k*NX*NY] = 0.00052 * 10000;		// unit: (kg*m/s^2) / m^2
					temperature[i + j*NX + k*NX*NY] = -53 + 273.15;		// unit: K
				}
			}
		}
	}

	for (i = 0; i < NX; i++) {
		for (j = 0; j < NY; j++) {
			for (k = 0; k < NZ; k++) {
				if (h_body[i + j*NX + k*NX*NY] != 0.0) {
					// air
					U[i + j*NX + k*NX*NY + 0 * N] = dens[i + j*NX + k*NX*NY];
					U[i + j*NX + k*NX*NY + 1 * N] = dens[i + j*NX + k*NX*NY] * (xv[i + j*NX + k*NX*NY]);
					U[i + j*NX + k*NX*NY + 2 * N] = dens[i + j*NX + k*NX*NY] * (yv[i + j*NX + k*NX*NY]);
					U[i + j*NX + k*NX*NY + 3 * N] = dens[i + j*NX + k*NX*NY] * (zv[i + j*NX + k*NX*NY]);
					U[i + j*NX + k*NX*NY + 4 * N] = dens[i + j*NX + k*NX*NY] * (CV*(press[i + j*NX + k*NX*NY] / dens[i + j*NX + k*NX*NY] / R)
						+ 0.5*((xv[i + j*NX + k*NX*NY] * xv[i + j*NX + k*NX*NY]) + (yv[i + j*NX + k*NX*NY] * yv[i + j*NX + k*NX*NY])
						+ (zv[i + j*NX + k*NX*NY] * zv[i + j*NX + k*NX*NY])));
				}
				else {
					// body
					for (z = 0; z < 5; z++) {
						U[i + j*NX + k*NX*NY + z * N] = 1.0;
					}
				}
			}
		}
	}
}

void Allocate_Memory() {
	size_t size = N*sizeof(float);
	cudaError_t Error;
	dens = (float*)malloc(size);
	xv = (float*)malloc(size);
	yv = (float*)malloc(size);
	zv = (float*)malloc(size);
	press = (float*)malloc(size);
	temperature = (float*)malloc(size);
	U = (float*)malloc(5 * size);
	Error = cudaMalloc((void**)&d_dens, size);
	printf("CUDA error (malloc d_dens) = %s\n",
		cudaGetErrorString(Error));
	Error = cudaMalloc((void**)&d_xv, size);
	printf("CUDA error (malloc d_xv) = %s\n",
		cudaGetErrorString(Error));
	Error = cudaMalloc((void**)&d_yv, size);
	printf("CUDA error (malloc d_yv) = %s\n",
		cudaGetErrorString(Error));
	Error = cudaMalloc((void**)&d_zv, size);
	printf("CUDA error (malloc d_zv) = %s\n",
		cudaGetErrorString(Error));
	Error = cudaMalloc((void**)&d_press, size);
	printf("CUDA error (malloc d_press) = %s\n",
		cudaGetErrorString(Error));
	Error = cudaMalloc((void**)&d_U, 5 * size);
	printf("CUDA error (malloc d_U) = %s\n",
		cudaGetErrorString(Error));
	Error = cudaMalloc((void**)&d_U_new, 5 * size);
	printf("CUDA error (malloc d_U_new) = %s\n",
		cudaGetErrorString(Error));
	Error = cudaMalloc((void**)&d_E, 5 * size);
	printf("CUDA error (malloc d_E) = %s\n",
		cudaGetErrorString(Error));
	Error = cudaMalloc((void**)&d_F, 5 * size);
	printf("CUDA error (malloc d_F) = %s\n",
		cudaGetErrorString(Error));
	Error = cudaMalloc((void**)&d_G, 5 * size);
	printf("CUDA error (malloc d_G) = %s\n",
		cudaGetErrorString(Error));
	Error = cudaMalloc((void**)&d_FR, 5 * size);
	printf("CUDA error (malloc d_FR) = %s\n",
		cudaGetErrorString(Error));
	Error = cudaMalloc((void**)&d_FL, 5 * size);
	printf("CUDA error (malloc d_FL) = %s\n",
		cudaGetErrorString(Error));
	Error = cudaMalloc((void**)&d_FU, 5 * size);
	printf("CUDA error (malloc d_FU) = %s\n",
		cudaGetErrorString(Error));
	Error = cudaMalloc((void**)&d_FD, 5 * size);
	printf("CUDA error (malloc d_FD) = %s\n",
		cudaGetErrorString(Error));
	Error = cudaMalloc((void**)&d_FF, 5 * size);
	printf("CUDA error (malloc d_FF) = %s\n",
		cudaGetErrorString(Error));
	Error = cudaMalloc((void**)&d_FB, 5 * size);
	printf("CUDA error (malloc d_FB) = %s\n",
		cudaGetErrorString(Error));
}

void Free_Memory() {
	if (dens) free(dens);
	if (xv) free(xv);
	if (yv) free(yv);
	if (zv) free(zv);
	if (press) free(press);
	if (temperature) free(temperature);
	if (d_dens) cudaFree(d_dens);
	if (d_xv) cudaFree(d_xv);
	if (d_yv) cudaFree(d_yv);
	if (d_zv) cudaFree(d_zv);
	if (d_U) cudaFree(d_U);
	if (d_U_new) cudaFree(d_U_new);
	if (d_E) cudaFree(d_E);
	if (d_F) cudaFree(d_F);
	if (d_G) cudaFree(d_G);
	if (d_FR) cudaFree(d_FR);
	if (d_FL) cudaFree(d_FL);
	if (d_FU) cudaFree(d_FU);
	if (d_FD) cudaFree(d_FD);
	if (d_FB) cudaFree(d_FB);
	if (d_FF) cudaFree(d_FF);

}

void Send_To_Device() {
	size_t size = N*sizeof(float);
	cudaError_t Error;
	Error = cudaMemcpy(d_body, h_body, size, cudaMemcpyHostToDevice);
	printf("CUDA error (memcpy h_body -> d_body) = %s\n", cudaGetErrorString(Error));
	Error = cudaMemcpy(d_dens, dens, size, cudaMemcpyHostToDevice);
	printf("CUDA error (memcpy dens -> d_dens) = %s\n", cudaGetErrorString(Error));
	Error = cudaMemcpy(d_xv, xv, size, cudaMemcpyHostToDevice);
	printf("CUDA error (memcpy xv -> d_xv) = %s\n", cudaGetErrorString(Error));
	Error = cudaMemcpy(d_yv, yv, size, cudaMemcpyHostToDevice);
	printf("CUDA error (memcpy yv -> d_yv) = %s\n", cudaGetErrorString(Error));
	Error = cudaMemcpy(d_zv, zv, size, cudaMemcpyHostToDevice);
	printf("CUDA error (memcpy zv -> d_zv) = %s\n", cudaGetErrorString(Error));
	Error = cudaMemcpy(d_press, press, size, cudaMemcpyHostToDevice);
	printf("CUDA error (memcpy press -> d_press) = %s\n", cudaGetErrorString(Error));
	Error = cudaMemcpy(d_U, U, 5 * size, cudaMemcpyHostToDevice);
	printf("CUDA error (memcpy U -> d_U) = %s\n", cudaGetErrorString(Error));

}

void Get_From_Device() {
	size_t size = N*sizeof(float);
	cudaError_t Error;
	Error = cudaMemcpy(dens, d_dens, size, cudaMemcpyDeviceToHost);
	printf("CUDA error (memcpy d_dens -> dens) = %s\n", cudaGetErrorString(Error));
	Error = cudaMemcpy(xv, d_xv, size, cudaMemcpyDeviceToHost);
	printf("CUDA error (memcpy d_xv -> xv) = %s\n", cudaGetErrorString(Error));
	Error = cudaMemcpy(yv, d_yv, size, cudaMemcpyDeviceToHost);
	printf("CUDA error (memcpy d_yv -> yv) = %s\n", cudaGetErrorString(Error));
	Error = cudaMemcpy(zv, d_zv, size, cudaMemcpyDeviceToHost);
	printf("CUDA error (memcpy d_zv -> zv) = %s\n", cudaGetErrorString(Error));
	Error = cudaMemcpy(press, d_press, size, cudaMemcpyDeviceToHost);
	printf("CUDA error (memcpy d_press -> press) = %s\n", cudaGetErrorString(Error));
}

__global__ void CalculateFlux(float* body, float* dens, float* xv, float* yv, float* zv, float* press,
	float* E, float* F, float* G,
	float* FL, float* FE, float* FB, float* FF, float* FD, float* FU,
	float* U, float* U_new) {
	float speed;
	int i = blockDim.x*blockIdx.x + threadIdx.x;

	if (i < N) {
		if (h_body[i + j*NX + k*NX*NY] == -1.0) { // air
			E[i + 0 * N] = dens[i] * xv[i];
			E[i + 1 * N] = dens[i] * xv[i] * xv[i] + press[i];
			E[i + 2 * N] = dens[i] * xv[i] * yv[i];
			E[i + 3 * N] = dens[i] * xv[i] * zv[i];
			E[i + 4 * N] = xv[i] * (U[i + 4 * N] + press[i]);

			F[i + 0 * N] = dens[i] * yv[i];
			F[i + 1 * N] = dens[i] * xv[i] * yv[i];
			F[i + 2 * N] = dens[i] * yv[i] * yv[i] + press[i];
			F[i + 3 * N] = dens[i] * yv[i] * zv[i];
			F[i + 4 * N] = yv[i] * (U[i + 4 * N] + press[i]);

			G[i + 0 * N] = dens[i] * zv[i];
			G[i + 1 * N] = dens[i] * xv[i] * zv[i];
			G[i + 2 * N] = dens[i] * yv[i] * zv[i];
			G[i + 3 * N] = dens[i] * zv[i] * zv[i] + press[i];
			G[i + 4 * N] = zv[i] * (U[i + 4 * N] + press[i]);
		}
	}
	__syncthreads();

	if (i < N) {
		if (h_body[i + j*NX + k*NX*NY] == -1.0) { // air
			// Rusanov flux: Left, Right, Back, Front, Down, Up
			if ((i % NX != 0) && (i % NX != (NX - 1)) && (i % (NX*NY) >= NX) && (i % (NX*NY) < NX*(NY - 1))) {
				speed = (float)((DX) / DT / 3.0)*0.5; //sqrt(GAMA*press[i] / dens[i]);		// speed of sound in air

				FL[i + 0 * N] = 0.5*(E[i + 0 * N] + E[i - 1 + 0 * N]) - speed*(U[i + 0 * N] - U[i - 1 + 0 * N]);
				FR[i + 0 * N] = 0.5*(E[i + 0 * N] + E[i + 1 + 0 * N]) - speed*(U[i + 1 + 0 * N] - U[i + 0 * N]);
				FL[i + 1 * N] = 0.5*(E[i + 1 * N] + E[i - 1 + 1 * N]) - speed*(U[i + 1 * N] - U[i - 1 + 1 * N]);
				FR[i + 1 * N] = 0.5*(E[i + 1 * N] + E[i + 1 + 1 * N]) - speed*(U[i + 1 + 1 * N] - U[i + 1 * N]);
				FL[i + 2 * N] = 0.5*(E[i + 2 * N] + E[i - 1 + 2 * N]) - speed*(U[i + 2 * N] - U[i - 1 + 2 * N]);
				FR[i + 2 * N] = 0.5*(E[i + 2 * N] + E[i + 1 + 2 * N]) - speed*(U[i + 1 + 2 * N] - U[i + 2 * N]);
				FL[i + 3 * N] = 0.5*(E[i + 3 * N] + E[i - 1 + 3 * N]) - speed*(U[i + 3 * N] - U[i - 1 + 3 * N]);
				FR[i + 3 * N] = 0.5*(E[i + 3 * N] + E[i + 1 + 3 * N]) - speed*(U[i + 1 + 3 * N] - U[i + 3 * N]);
				FL[i + 4 * N] = 0.5*(E[i + 4 * N] + E[i - 1 + 4 * N]) - speed*(U[i + 4 * N] - U[i - 1 + 4 * N]);
				FR[i + 4 * N] = 0.5*(E[i + 4 * N] + E[i + 1 + 4 * N]) - speed*(U[i + 1 + 4 * N] - U[i + 4 * N]);

				FB[i + 0 * N] = 0.5*(F[i + 0 * N] + F[i - NX + 0 * N]) - speed*(U[i + 0 * N] - U[i - NX + 0 * N]);
				FF[i + 0 * N] = 0.5*(F[i + 0 * N] + F[i + NX + 0 * N]) - speed*(U[i + NX + 0 * N] - U[i + 0 * N]);
				FB[i + 1 * N] = 0.5*(F[i + 1 * N] + F[i - NX + 1 * N]) - speed*(U[i + 1 * N] - U[i - NX + 1 * N]);
				FF[i + 1 * N] = 0.5*(F[i + 1 * N] + F[i + NX + 1 * N]) - speed*(U[i + NX + 1 * N] - U[i + 1 * N]);
				FB[i + 2 * N] = 0.5*(F[i + 2 * N] + F[i - NX + 2 * N]) - speed*(U[i + 2 * N] - U[i - NX + 2 * N]);
				FF[i + 2 * N] = 0.5*(F[i + 2 * N] + F[i + NX + 2 * N]) - speed*(U[i + NX + 2 * N] - U[i + 2 * N]);
				FB[i + 3 * N] = 0.5*(F[i + 3 * N] + F[i - NX + 3 * N]) - speed*(U[i + 3 * N] - U[i - NX + 3 * N]);
				FF[i + 3 * N] = 0.5*(F[i + 3 * N] + F[i + NX + 3 * N]) - speed*(U[i + NX + 3 * N] - U[i + 3 * N]);
				FB[i + 4 * N] = 0.5*(F[i + 4 * N] + F[i - NX + 4 * N]) - speed*(U[i + 4 * N] - U[i - NX + 4 * N]);
				FF[i + 4 * N] = 0.5*(F[i + 4 * N] + F[i + NX + 4 * N]) - speed*(U[i + NX + 4 * N] - U[i + 4 * N]);

				FD[i + 0 * N] = 0.5*(G[i + 0 * N] + G[i - NX*NY + 0 * N]) - speed*(U[i + 0 * N] - U[i - NX*NY + 0 * N]);
				FU[i + 0 * N] = 0.5*(G[i + 0 * N] + G[i + NX*NY + 0 * N]) - speed*(U[i + NX*NY + 0 * N] - U[i + 0 * N]);
				FD[i + 1 * N] = 0.5*(G[i + 1 * N] + G[i - NX*NY + 1 * N]) - speed*(U[i + 1 * N] - U[i - NX*NY + 1 * N]);
				FU[i + 1 * N] = 0.5*(G[i + 1 * N] + G[i + NX*NY + 1 * N]) - speed*(U[i + NX*NY + 1 * N] - U[i + 1 * N]);
				FD[i + 2 * N] = 0.5*(G[i + 2 * N] + G[i - NX*NY + 2 * N]) - speed*(U[i + 2 * N] - U[i - NX*NY + 2 * N]);
				FU[i + 2 * N] = 0.5*(G[i + 2 * N] + G[i + NX*NY + 2 * N]) - speed*(U[i + NX*NY + 2 * N] - U[i + 2 * N]);
				FD[i + 3 * N] = 0.5*(G[i + 3 * N] + G[i - NX*NY + 3 * N]) - speed*(U[i + 3 * N] - U[i - NX*NY + 3 * N]);
				FU[i + 3 * N] = 0.5*(G[i + 3 * N] + G[i + NX*NY + 3 * N]) - speed*(U[i + NX*NY + 3 * N] - U[i + 3 * N]);
				FD[i + 4 * N] = 0.5*(G[i + 4 * N] + G[i - NX*NY + 4 * N]) - speed*(U[i + 4 * N] - U[i - NX*NY + 4 * N]);
				FU[i + 4 * N] = 0.5*(G[i + 4 * N] + G[i + NX*NY + 4 * N]) - speed*(U[i + NX*NY + 4 * N] - U[i + 4 * N]);
			}
		}
	}
	__syncthreads();

	if (i < N) {
		// revise body condition when it is near air
		if (h_body[(i - 1) + j*NX + k*NX*NY] == 0.0) { // left is body

			E[(i - 1) + j*NX + k*NX*NY + 0 * N] = -E[(i)+j*NX + k*NX*NY + 0 * N];
			U[(i - 1) + j*NX + k*NX*NY + 0 * N] = U[(i)+j*NX + k*NX*NY + 0 * N];
			E[(i - 1) + j*NX + k*NX*NY + 4 * N] = -E[(i)+j*NX + k*NX*NY + 4 * N];
			U[(i - 1) + j*NX + k*NX*NY + 4 * N] = U[(i)+j*NX + k*NX*NY + 4 * N];

			E[(i - 1) + j*NX + k*NX*NY + z*N] = E[(i)+j*NX + k*NX*NY + z*N];
			U[(i - 1) + j*NX + k*NX*NY + z*N] = -U[(i)+j*NX + k*NX*NY + z*N];

			FL[i + j*NX + k*NX*NY + z*N] = 0.5*(E[i + j*NX + k*NX*NY + z*N] + E[(i - 1) + j*NX + k*NX*NY + z*N])
				- speed*(U[i + j*NX + k*NX*NY + z*N] - U[(i - 1) + j*NX + k*NX*NY + z*N]);
		}

		if (h_body[(i + 1) + j*NX + k*NX*NY] == 0.0) {
			for (z = 0; z < 5; z++) {
				if (z == 0 || z == 4){
					E[(i + 1) + j*NX + k*NX*NY + z*N] = -E[(i)+j*NX + k*NX*NY + z*N];
					U[(i + 1) + j*NX + k*NX*NY + z*N] = U[(i)+j*NX + k*NX*NY + z*N];
				}
				else{
					E[(i + 1) + j*NX + k*NX*NY + z*N] = E[(i)+j*NX + k*NX*NY + z*N];
					U[(i + 1) + j*NX + k*NX*NY + z*N] = -U[(i)+j*NX + k*NX*NY + z*N];
				}
				FR[i + j*NX + k*NX*NY + z*N] = 0.5*(E[i + j*NX + k*NX*NY + z*N] + E[(i + 1) + j*NX + k*NX*NY + z*N])
					- speed*(U[i + 1 + j*NX + k*NX*NY + z*N] - U[(i)+j*NX + k*NX*NY + z*N]);
			}
		}
		if (h_body[i + (j - 1)*NX + k*NX*NY] == 0.0) {
			for (z = 0; z < 5; z++) {
				if (z == 0 || z == 4){
					F[i + (j - 1)*NX + k*NX*NY + z*N] = -F[(i)+j*NX + k*NX*NY + z*N];
					U[i + (j - 1)*NX + k*NX*NY + z*N] = U[(i)+j*NX + k*NX*NY + z*N];
				}
				else{
					F[i + (j - 1)*NX + k*NX*NY + z*N] = F[(i)+j*NX + k*NX*NY + z*N];
					U[i + (j - 1)*NX + k*NX*NY + z*N] = -U[(i)+j*NX + k*NX*NY + z*N];
				}
				FB[i + j*NX + k*NX*NY + z*N] = 0.5*(F[i + j*NX + k*NX*NY + z*N] + F[i + (j - 1)*NX + k*NX*NY + z*N])
					- speed*(U[i + j*NX + k*NX*NY + z*N] - U[i + (j - 1)*NX + k*NX*NY + z*N]);
			}
		}
		if (h_body[i + (j + 1)*NX + k*NX*NY] == 0.0) {
			for (z = 0; z < 5; z++) {
				if (z == 0 || z == 4){
					F[i + (j + 1)*NX + k*NX*NY + z*N] = -F[(i)+j*NX + k*NX*NY + z*N];
					U[i + (j + 1)*NX + k*NX*NY + z*N] = U[(i)+j*NX + k*NX*NY + z*N];
				}
				else{
					F[i + (j + 1)*NX + k*NX*NY + z*N] = F[(i)+j*NX + k*NX*NY + z*N];
					U[i + (j + 1)*NX + k*NX*NY + z*N] = -U[(i)+j*NX + k*NX*NY + z*N];
				}
				FF[i + j*NX + k*NX*NY + z*N] = 0.5*(F[i + j*NX + k*NX*NY + z*N] + F[i + (j + 1)*NX + k*NX*NY + z*N])
					- speed*(U[i + (j + 1)*NX + k*NX*NY + z*N] - U[i + (j)*NX + k*NX*NY + z*N]);
			}
		}
		if (h_body[i + j*NX + (k - 1)*NX*NY] == 0.0) {
			for (z = 0; z < 5; z++) {
				if (z == 0 || z == 4){
					G[i + j*NX + (k - 1)*NX*NY + z*N] = -G[(i)+j*NX + k*NX*NY + z*N];
					U[i + j*NX + (k - 1)*NX*NY + z*N] = U[(i)+j*NX + k*NX*NY + z*N];
				}
				else{
					G[i + j*NX + (k - 1)*NX*NY + z*N] = G[(i)+j*NX + k*NX*NY + z*N];
					U[i + j*NX + (k - 1)*NX*NY + z*N] = -U[(i)+j*NX + k*NX*NY + z*N];
				}
				FD[i + j*NX + k*NX*NY + z*N] = 0.5*(G[i + j*NX + k*NX*NY + z*N] + G[i + j*NX + (k - 1)*NX*NY + z*N])
					- speed*(U[i + j*NX + k*NX*NY + z*N] - U[i + j*NX + (k - 1)*NX*NY + z*N]);
			}
		}
		if (h_body[i + j*NX + (k + 1)*NX*NY] == 0.0) {
			for (z = 0; z < 5; z++) {
				if (z == 0 || z == 4){
					G[i + j*NX + (k + 1)*NX*NY + z*N] = -G[(i)+j*NX + k*NX*NY + z*N];
					U[i + j*NX + (k + 1)*NX*NY + z*N] = U[(i)+j*NX + k*NX*NY + z*N];
				}
				else{
					G[i + j*NX + (k + 1)*NX*NY + z*N] = G[(i)+j*NX + k*NX*NY + z*N];
					U[i + j*NX + (k + 1)*NX*NY + z*N] = -U[(i)+j*NX + k*NX*NY + z*N];
				}
				FU[i + j*NX + k*NX*NY + z*N] = 0.5*(G[i + j*NX + k*NX*NY + z*N] + G[i + j*NX + (k + 1)*NX*NY + z*N])
					- speed*(U[i + j*NX + (k + 1)*NX*NY + z*N] - U[i + j*NX + (k)*NX*NY + z*N]);
			}
		}
	}
	__syncthreads();

	// Update U by U_new using FVM (Rusanov Flux)
	if (i < N) {
		if ((i % NX != 0) && (i % NX != (NX - 1)) && (i % (NX*NY) >= NX) && (i % (NX*NY) < NX*(NY - 1))) {
			U_new[i + 0 * N] = U[i + 0 * N] - (DT / DX)*(FR[i + 0 * N] - FL[i + 0 * N])
				- (DT / DY)*(FF[i + 0 * N] - FB[i + 0 * N]) - (DT / DZ)*(FU[i + 0 * N] - FD[i + 0 * N]);
			U_new[i + 1 * N] = U[i + 1 * N] - (DT / DX)*(FR[i + 1 * N] - FL[i + 1 * N])
				- (DT / DY)*(FF[i + 1 * N] - FB[i + 1 * N]) - (DT / DZ)*(FU[i + 1 * N] - FD[i + 1 * N]);
			U_new[i + 2 * N] = U[i + 2 * N] - (DT / DX)*(FR[i + 2 * N] - FL[i + 2 * N])
				- (DT / DY)*(FF[i + 2 * N] - FB[i + 2 * N]) - (DT / DZ)*(FU[i + 2 * N] - FD[i + 2 * N]);
			U_new[i + 3 * N] = U[i + 3 * N] - (DT / DX)*(FR[i + 3 * N] - FL[i + 3 * N])
				- (DT / DY)*(FF[i + 3 * N] - FB[i + 3 * N]) - (DT / DZ)*(FU[i + 3 * N] - FD[i + 3 * N]);
			U_new[i + 4 * N] = U[i + 4 * N] - (DT / DX)*(FR[i + 4 * N] - FL[i + 4 * N])
				- (DT / DY)*(FF[i + 4 * N] - FB[i + 4 * N]) - (DT / DZ)*(FU[i + 4 * N] - FD[i + 4 * N]);
		}

		//Renew back and front boundary condition
		else if (i % (NX*NY) >= 1 && i % (NX*NY) < (NX - 1)) {
			// U_new[i] of back boundary = U_new[i+NX]
			U_new[i + 0 * N] = U_new[i + NX + 0 * N];
			U_new[i + 1 * N] = U_new[i + NX + 1 * N];
			U_new[i + 2 * N] = U_new[i + NX + 2 * N];
			U_new[i + 3 * N] = U_new[i + NX + 3 * N];
			U_new[i + 4 * N] = U_new[i + NX + 4 * N];
			// U_new[i] of front boundary = U_new[i-NX]
			U_new[i + (NY - 1)*NX + 0 * N] = U_new[i + (NY - 2)*NX + 0 * N];
			U_new[i + (NY - 1)*NX + 1 * N] = U_new[i + (NY - 2)*NX + 1 * N];
			U_new[i + (NY - 1)*NX + 2 * N] = U_new[i + (NY - 2)*NX + 2 * N];
			U_new[i + (NY - 1)*NX + 3 * N] = U_new[i + (NY - 2)*NX + 3 * N];
			U_new[i + (NY - 1)*NX + 4 * N] = U_new[i + (NY - 2)*NX + 4 * N];
		}

		//Renew left and right boundary condition
		else if (i % (NX*NY) >= 0 && i % (NX*NY) < NY) {
			// U_new[i] of left boundary = U_new[i+1]
			U_new[i*NX + 0 * N] = U_new[i*NX + 1 + 0 * N];
			U_new[i*NX + 1 * N] = U_new[i*NX + 1 + 1 * N];
			U_new[i*NX + 2 * N] = U_new[i*NX + 1 + 2 * N];
			U_new[i*NX + 3 * N] = U_new[i*NX + 1 + 3 * N];
			U_new[i*NX + 4 * N] = U_new[i*NX + 1 + 4 * N];
			// U_new[i] of right boundary = U_new[i-1]
			U_new[i*NX + (NX - 1) + 0 * N] = U_new[i*NX + (NX - 2) + 0 * N];
			U_new[i*NX + (NX - 1) + 1 * N] = U_new[i*NX + (NX - 2) + 1 * N];
			U_new[i*NX + (NX - 1) + 2 * N] = U_new[i*NX + (NX - 2) + 2 * N];
			U_new[i*NX + (NX - 1) + 3 * N] = U_new[i*NX + (NX - 2) + 3 * N];
			U_new[i*NX + (NX - 1) + 4 * N] = U_new[i*NX + (NX - 2) + 4 * N];
		}
	}
	__syncthreads();

	// Update density, velocity, pressure, and U
	if (i < N) {
		dens[i] = U_new[i + 0 * N];
		xv[i] = U_new[i + 1 * N] / dens[i];
		yv[i] = U_new[i + 2 * N] / dens[i];
		zv[i] = U_new[i + 3 * N] / dens[i];
		press[i] = (GAMA - 1) * (U_new[i + 4 * N] - 0.5 * dens[i] * (xv[i] * xv[i] + yv[i] * yv[i] + zv[i] * zv[i]));
		U[i + 0 * N] = U_new[i + 0 * N];
		U[i + 1 * N] = U_new[i + 1 * N];
		U[i + 2 * N] = U_new[i + 2 * N];
		U[i + 3 * N] = U_new[i + 3 * N];
		U[i + 4 * N] = U_new[i + 4 * N];
	}
	__syncthreads();
}


void Call_CalculateFlux() {
	int threadsPerBlock = 256;
	int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
	CalculateFlux<<<blocksPerGrid, threadsPerBlock>>>(d_body, d_dens, d_xv, d_yv, d_zv, d_press, d_E, d_F, d_G, d_FL, d_FR, d_FB, d_FF, d_FD, d_FU, d_U, d_U_new);
}

void Save_Data_To_File(char *output_file_name) {
	FILE *pOutPutFile;
	pOutPutFile = fopen(output_file_name, "w");
	if (!pOutPutFile) { printf("Open failure"); }

	fprintf(pOutPutFile, "TITLE=\"Flow Field of X-37\"\n");
	fprintf(pOutPutFile, "VARIABLES=\"X\", \"Y\", \"Z\", \"U\", \"V\", \"W\", \"Pressure\", \"Temperature\", \"Body\"\n");
	fprintf(pOutPutFile, "ZONE I = 100, J = 100, K = 100, F = POINT\n");

	int i, j, k;
	for (k = 0; k < NZ; k++) {
		for (j = 0; j < NY; j++) {
			for (i = 0; i < NX; i++) {
				temperature[i + j*NX + k*NX*NY] = press[i + j*NX + k*NX*NY] / (dens[i + j*NX + k*NX*NY] * R);
				/* ...test body...*/
				//fprintf(pOutPutFile, "%d %d %d %f\n", i, j, k, h_body[i + j*NX + k*NX*NY]);
				/* ...test body...*/
				fprintf(pOutPutFile, "%d %d %d %f %f %f %f %f\n", i, j, k, xv[i + j*NX + k*NX*NY], yv[i + j*NX + k*NX*NY], zv[i + j*NX + k*NX*NY], press[i + j*NX + k*NX*NY], temperature[i + j*NX + k*NX*NY], h_body[i + j*NX + k*NX*NY]);
			}
		}
	}
}
