#include "gpumain.h"
float *dens;              //density
float *xv;                //velocity in x
float *yv;                //velocity in y
float *zv;
float *temperature;
float *press;             //pressure
float *d_dens;              //density
float *d_xv;                //velocity in x
float *d_yv;                //velocity in y
float *d_zv;
float *d_temperature;        //temprature
float *d_press;             //pressure
float *U;
float *d_FF;
float *d_FB;
float *d_FR;
float *d_FL;
float *d_FU;
float *d_FD;
float *d_U;
float *d_U_new;
float *d_E;
float *d_F;
float *d_G;

float *h_body;
float *d_body;

__global__ void CalculateFlux(float* dens,float* xv,float* yv,float* press,
float* E,float* F,float* FR,float* FL,float* FU,float* FD,float* U,float* U_new);

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
				if (h_body[i + j*NX + k*NX*NY] == 0.0) {
					// body
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


void Allocate_Memory(){
	size_t size = N*sizeof(float);
	cudaError_t Error;
	dens = (float*)malloc(size);
	xv = (float*)malloc(size);
	yv = (float*)malloc(size);
	zv = (float*)malloc(size);
	press = (float*)malloc(size);
	temperature = (float*)malloc(size);
	U=(float*)malloc(5*size); 
	Error = cudaMalloc((void**)&d_dens,size); 
                printf("CUDA error (malloc d_dens) = %s\n",    
                cudaGetErrorString(Error));
	Error = cudaMalloc((void**)&d_xv,size);
                printf("CUDA error (malloc d_xv) = %s\n",
                cudaGetErrorString(Error));
	Error = cudaMalloc((void**)&d_yv,size);
                printf("CUDA error (malloc d_yv) = %s\n",
                cudaGetErrorString(Error));
	Error = cudaMalloc((void**)&d_zv, size);
				printf("CUDA error (malloc d_yv) = %s\n",
					cudaGetErrorString(Error));
	Error = cudaMalloc((void**)&d_press,size);
                printf("CUDA error (malloc d_press) = %s\n",
                cudaGetErrorString(Error));
	Error = cudaMalloc((void**)&d_temperature, size);
				printf("CUDA error (malloc d_press) = %s\n",
					cudaGetErrorString(Error));
	Error = cudaMalloc((void**)&d_U,5*size);
                printf("CUDA error (malloc d_U) = %s\n",
                cudaGetErrorString(Error));
	Error = cudaMalloc((void**)&d_U_new,5*size);
                printf("CUDA error (malloc d_U_new) = %s\n",
                cudaGetErrorString(Error));
	Error = cudaMalloc((void**)&d_E,5*size);
                printf("CUDA error (malloc d_E) = %s\n",
                cudaGetErrorString(Error));
	Error = cudaMalloc((void**)&d_F,5*size);
                printf("CUDA error (malloc d_F) = %s\n",
                cudaGetErrorString(Error));
	Error = cudaMalloc((void**)&d_G, 5 * size);
				printf("CUDA error (malloc d_F) = %s\n",
					cudaGetErrorString(Error));
	Error = cudaMalloc((void**)&d_FR,5*size);
                printf("CUDA error (malloc d_FR) = %s\n",
                cudaGetErrorString(Error));
	Error = cudaMalloc((void**)&d_FL,5*size);
                printf("CUDA error (malloc d_FL) = %s\n",
                cudaGetErrorString(Error));
	Error = cudaMalloc((void**)&d_FU,5*size);
                printf("CUDA error (malloc d_FU) = %s\n",
                cudaGetErrorString(Error));
	Error = cudaMalloc((void**)&d_FD,5*size);
                printf("CUDA error (malloc d_FD) = %s\n",
                cudaGetErrorString(Error));
	Error = cudaMalloc((void**)&d_FF, 5* size);
				printf("CUDA error (malloc d_FD) = %s\n",
					cudaGetErrorString(Error));
	Error = cudaMalloc((void**)&d_FB, 5 * size);
				printf("CUDA error (malloc d_FD) = %s\n",
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
	if (d_temperature) free(d_temperature);
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

void Send_To_Device()
{
size_t size=N*sizeof(float);
cudaError_t Error;
Error = cudaMemcpy(d_body, h_body, size, cudaMemcpyHostToDevice);
printf("CUDA error (memcpy h_body -> d_body) = %s\n", cudaGetErrorString(Error));
Error=cudaMemcpy(d_dens,dens,size,cudaMemcpyHostToDevice);
printf("CUDA error (memcpy dens -> d_dens) = %s\n",cudaGetErrorString(Error));
Error=cudaMemcpy(d_xv,xv,size,cudaMemcpyHostToDevice);
printf("CUDA error (memcpy xv -> d_xv) = %s\n",cudaGetErrorString(Error));
Error=cudaMemcpy(d_yv,yv,size,cudaMemcpyHostToDevice);
printf("CUDA error (memcpy yv -> d_yv) = %s\n",cudaGetErrorString(Error));
Error = cudaMemcpy(d_zv, zv, size, cudaMemcpyHostToDevice);
printf("CUDA error (memcpy zv -> d_zv) = %s\n", cudaGetErrorString(Error));
Error=cudaMemcpy(d_press,press,size,cudaMemcpyHostToDevice);
printf("CUDA error (memcpy press -> d_press) = %s\n",cudaGetErrorString(Error));
Error=cudaMemcpy(d_U,U,5*size,cudaMemcpyHostToDevice);
printf("CUDA error (memcpy U -> d_U) = %s\n",cudaGetErrorString(Error));

}

void Get_From_Device()
{
size_t size=N*sizeof(float);
cudaError_t Error;
Error=cudaMemcpy(dens,d_dens,size,cudaMemcpyDeviceToHost);
printf("CUDA error (memcpy d_dens -> dens) = %s\n",cudaGetErrorString(Error));
Error=cudaMemcpy(xv,d_xv,size,cudaMemcpyDeviceToHost);
printf("CUDA error (memcpy d_xv -> xv) = %s\n",cudaGetErrorString(Error));
Error=cudaMemcpy(yv,d_yv,size,cudaMemcpyDeviceToHost);
printf("CUDA error (memcpy d_yv -> yv) = %s\n",cudaGetErrorString(Error));
Error = cudaMemcpy(zv, d_zv, size, cudaMemcpyDeviceToHost);
printf("CUDA error (memcpy d_zv -> zv) = %s\n", cudaGetErrorString(Error));
Error=cudaMemcpy(press,d_press,size,cudaMemcpyDeviceToHost);
printf("CUDA error (memcpy d_press -> press) = %s\n",cudaGetErrorString(Error));

}

__global__ void CalculateFlux(float* dens,float* xv,float* yv,float* press,
float* E,float* F,float* FR,float* FL,float* FU,float* FD,float* U,float* U_new) {
	

		
}


void Call_CalculateFlux(){
int threadsPerBlock=256;
int blocksPerGrid=(N+threadsPerBlock-1)/threadsPerBlock;
CalculateFlux<<<blocksPerGrid,threadsPerBlock>>>(d_dens,d_xv,d_yv,d_press,d_E,d_F,d_FR,d_FL,d_FU,d_FD,d_U,d_U_new);
}

void Save_Data_To_File(char *output_file_name) {
	FILE *pOutPutFile;
	pOutPutFile = fopen(output_file_name, "w");
	if (!pOutPutFile) { printf("Open failure"); }

	fprintf(pOutPutFile, "TITLE=\"Flow Field of X-37\"\n");
	/* ...test body...*/
	//fprintf(pOutPutFile, "VARIABLES=\"X\", \"Y\", \"Z\", \"Body\"\n");
	/* ...test body...*/
	fprintf(pOutPutFile, "VARIABLES=\"X\", \"Y\", \"Z\", \"U\", \"V\", \"W\", \"Pressure\", \"Temperature\"\n");
	fprintf(pOutPutFile, "ZONE I = 100, J = 100, K = 100, F = POINT\n");

	int i, j, k;
	for (k = 0; k < NZ; k++) {
		for (j = 0; j < NY; j++) {
			for (i = 0; i < NX; i++) {
				temperature[i + j*NX + k*NX*NY] = press[i + j*NX + k*NX*NY] / (dens[i + j*NX + k*NX*NY] * R);
				/* ...test body...*/
				//fprintf(pOutPutFile, "%d %d %d %f\n", i, j, k, h_body[i + j*NX + k*NX*NY]);
				/* ...test body...*/
				fprintf(pOutPutFile, "%d %d %d %f %f %f %f %f\n", i, j, k, xv[i + j*NX + k*NX*NY], yv[i + j*NX + k*NX*NY], zv[i + j*NX + k*NX*NY], press[i + j*NX + k*NX*NY], temperature[i + j*NX + k*NX*NY]);
			}
		}
	}
}
	

