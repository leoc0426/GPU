#include "gpumain.h"
float *dens;              //density
float *xv;                //velocity in x
float *yv;                //velocity in y
float *press;             //pressure
float *d_dens;              //density
float *d_xv;                //velocity in x
float *d_yv;                //velocity in y
float *d_press;             //pressure
float *U;
float *U_new;
float *E;
float *F;
float *FR;
float *FL;
float *FU;
float *FD;
float *d_U;
float *d_U_new;
float *d_E;
float *d_F;
float *d_FR;
float *d_FL;
float *d_FU;
float *d_FD;

__global__ void CalculateFlux(float* dens,float* xv,float* yv,float* press,
float* E,float* F,float* FR,float* FL,float* FU,float* FD,float* U,float* U_new);

void Init() {
        int i, j;
        for (j = 0; j < NY; j++) {
                for (i = 0; i < NX; i++) {
                        float d =0;
                        float cx = dx*(i+0.5);
                        float cy = dy*(j+0.5);
                        if (i < 0.1*NX) {
                                //Initialize the right side gas condition
                                dens[i + j*NX] = 3.81;
                                xv[i + j*NX] = 0.0;
                                yv[i + j*NX] = 0.0;
                                press[i + j*NX] = 10.0;
                        } else {
                                d = (cx - 0.4)*(cx - 0.4) + cy* cy;
                                if(d <= (0.04*(L*L)))
                                {
                                        //Initialize the left side gas condition
                                        dens[i + j*NX] = 0.1;
                                        xv[i + j*NX] = 0.0;
                                        yv[i + j*NX] = 0.0;
                                        press[i + j*NX] = 10.0;

                                } else {
                                        //Initialize the left side gas condition
                                        dens[i + j*NX] = 1.0;
                                        xv[i + j*NX] = 0.0;
                                        yv[i + j*NX] = 0.0;
                                        press[i + j*NX] = 1.0;
                                }
                        }
                        U[i+j*NX] = dens[i+j*NX];
                        U[i+j*NX+NX*NY] = dens[i+j*NX] * (xv[i+j*NX]);
                        U[i+j*NX+2*NX*NY] = dens[i+j*NX] * (yv[i+j*NX]);
                        U[i+j*NX+3*NX*NY] = dens[i+j*NX] * (CV*(press[i+j*NX]/dens[i+j*NX]/R)
                                + 0.5*((xv[i+j*NX] * xv[i+j*NX]) + (yv[i+j*NX] * yv[i+j*NX])));
                }
        }
}


void Allocate_Memory(){
	size_t size = N*sizeof(float);
	cudaError_t Error;
	dens = (float*)malloc(size);
	xv=(float*)malloc(size);
	yv = (float*)malloc(size);
	press = (float*)malloc(size);
	U=(float*)malloc(4*size); 
	Error = cudaMalloc((void**)&d_dens,size); 
                printf("CUDA error (malloc d_dens) = %s\n",    
                cudaGetErrorString(Error));
	Error = cudaMalloc((void**)&d_xv,size);
                printf("CUDA error (malloc d_xv) = %s\n",
                cudaGetErrorString(Error));
	Error = cudaMalloc((void**)&d_yv,size);
                printf("CUDA error (malloc d_yv) = %s\n",
                cudaGetErrorString(Error));
	Error = cudaMalloc((void**)&d_press,size);
                printf("CUDA error (malloc d_press) = %s\n",
                cudaGetErrorString(Error));
	Error = cudaMalloc((void**)&d_U,4*size);
                printf("CUDA error (malloc d_U) = %s\n",
                cudaGetErrorString(Error));
	Error = cudaMalloc((void**)&d_U_new,4*size);
                printf("CUDA error (malloc d_U_new) = %s\n",
                cudaGetErrorString(Error));
	Error = cudaMalloc((void**)&d_E,4*size);
                printf("CUDA error (malloc d_E) = %s\n",
                cudaGetErrorString(Error));
	Error = cudaMalloc((void**)&d_F,4*size);
                printf("CUDA error (malloc d_F) = %s\n",
                cudaGetErrorString(Error));
	Error = cudaMalloc((void**)&d_FR,4*size);
                printf("CUDA error (malloc d_FR) = %s\n",
                cudaGetErrorString(Error));
	Error = cudaMalloc((void**)&d_FL,4*size);
                printf("CUDA error (malloc d_FL) = %s\n",
                cudaGetErrorString(Error));
	Error = cudaMalloc((void**)&d_FU,4*size);
                printf("CUDA error (malloc d_FU) = %s\n",
                cudaGetErrorString(Error));
	Error = cudaMalloc((void**)&d_FD,4*size);
                printf("CUDA error (malloc d_FD) = %s\n",
                cudaGetErrorString(Error));

}

void Free_Memory() {
	if (dens) free(dens);
	if (xv) free(xv);
	if (yv) free(yv);
	if (press) free(press);
	if (d_dens) cudaFree(d_dens);
	if (d_xv) cudaFree(d_xv);
	if (d_yv) cudaFree(d_yv);
	if (d_U) cudaFree(d_U);
        if (d_U_new) cudaFree(d_U_new);
	if (d_E) cudaFree(d_E);
        if (d_F) cudaFree(d_F);
	if (d_FR) cudaFree(d_FR);
        if (d_FL) cudaFree(d_FL);
	if (d_FU) cudaFree(d_FU);
        if (d_FD) cudaFree(d_FD);

}

void Send_To_Device()
{
size_t size=N*sizeof(float);
cudaError_t Error;
Error=cudaMemcpy(d_dens,dens,size,cudaMemcpyHostToDevice);
printf("CUDA error (memcpy dens -> d_dens) = %s\n",cudaGetErrorString(Error));
Error=cudaMemcpy(d_xv,xv,size,cudaMemcpyHostToDevice);
printf("CUDA error (memcpy xv -> d_xv) = %s\n",cudaGetErrorString(Error));
Error=cudaMemcpy(d_yv,yv,size,cudaMemcpyHostToDevice);
printf("CUDA error (memcpy yv -> d_yv) = %s\n",cudaGetErrorString(Error));
Error=cudaMemcpy(d_press,press,size,cudaMemcpyHostToDevice);
printf("CUDA error (memcpy press -> d_press) = %s\n",cudaGetErrorString(Error));
Error=cudaMemcpy(d_U,U,4*size,cudaMemcpyHostToDevice);
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
Error=cudaMemcpy(press,d_press,size,cudaMemcpyDeviceToHost);
printf("CUDA error (memcpy d_press -> press) = %s\n",cudaGetErrorString(Error));

}

__global__ void CalculateFlux(float* dens,float* xv,float* yv,float* press,
float* E,float* F,float* FR,float* FL,float* FU,float* FD,float* U,float* U_new) {
	float speed;
	int i=blockDim.x*blockIdx.x+threadIdx.x;

	if(i<N){
			E[i] = dens[i]*xv[i];
			E[i+NX*NY] = dens[i]*xv[i]*xv[i] + press[i];
			E[i+2*NX*NY] = dens[i]*xv[i]*yv[i];
			E[i+3*NX*NY] = xv[i] * (U[i+3*NX*NY] + press[i]);
			
			F[i] = dens[i]*yv[i];
			F[i+NX*NY] = dens[i]*xv[i]*yv[i];
			F[i+2*NX*NY] = dens[i]*yv[i]*yv[i] + press[i];
			F[i+3*NX*NY] = yv[i] * (U[i+3*NX*NY] + press[i]);
		}	
// Rusanov flux:Left, Right, Up, Down

__syncthreads();
		if(i%NX!=0 && i%NX!=(NX-1) && i>=NX && i<NX*(NY-1)){
			speed = sqrt(GAMA*press[i]/dens[i]);		// speed of sound in air
			
			FL[i] = 0.5*(E[i] + E[i-1]) - speed*(U[i] - U[i-1]);
			FR[i] = 0.5*(E[i] + E[i+1]) - speed*(U[i+1] - U[i]);
			FL[i+NX*NY] = 0.5*(E[i+NX*NY] + E[i+NX*NY-1]) - speed*(U[i+NX*NY] - U[i+NX*NY-1]);
			FR[i+NX*NY] = 0.5*(E[i+NX*NY] + E[i+NX*NY+1]) - speed*(U[i+NX*NY+1] - U[i+NX*NY]);
			FL[i+2*NX*NY] = 0.5*(E[i+2*NX*NY] + E[i+2*NX*NY-1]) - speed*(U[i+2*NX*NY] - U[i+2*NX*NY-1]);
			FR[i+2*NX*NY] = 0.5*(E[i+2*NX*NY] + E[i+2*NX*NY+1]) - speed*(U[i+2*NX*NY+1] - U[i+2*NX*NY]);
			FL[i+3*NX*NY] = 0.5*(E[i+3*NX*NY] + E[i+3*NX*NY-1]) - speed*(U[i+3*NX*NY] - U[i+3*NX*NY-1]);
			FR[i+3*NX*NY] = 0.5*(E[i+3*NX*NY] + E[i+3*NX*NY+1]) - speed*(U[i+3*NX*NY+1] - U[i+3*NX*NY]);

			FD[i] = 0.5*(F[i-NX] + F[i])- speed*(U[i] -U[i-NX]);
			FU[i] = 0.5*(F[i] + F[i+NX])- speed*(U[i+NX] - U[i]);
			FD[i+NX*NY] = 0.5*(F[i-NX+NX*NY] + F[i+NX*NY])- speed*(U[i+NX*NY] - U[i-NX+NX*NY]);
			FU[i+NX*NY] = 0.5*(F[i+NX*NY] + F[i+NX+NX*NY])- speed*(U[i+NX+NX*NY] - U[i+NX*NY]);
			FD[i+2*NX*NY] = 0.5*(F[i-NX+2*NX*NY] + F[i+2*NX*NY])- speed*(U[i+2*NX*NY] - U[i-NX+2*NX*NY]);
			FU[i+2*NX*NY] = 0.5*(F[i+2*NX*NY] + F[i+NX+2*NX*NY])- speed*(U[i+NX+2*NX*NY] - U[i+2*NX*NY]);
			FD[i+3*NX*NY] = 0.5*(F[i-NX+3*NX*NY] + F[i+3*NX*NY])- speed*(U[i+3*NX*NY] - U[i-NX+3*NX*NY]);
			FU[i+3*NX*NY] = 0.5*(F[i+3*NX*NY] + F[i+NX+3*NX*NY])- speed*(U[i+NX+3*NX*NY] - U[i+3*NX*NY]);
		}
	
__syncthreads();

// Update U by FVM
        if(i%NX!=0 && i%NX!=(NX-1) && i>=NX && i<NX*(NY-1)){
                        U_new[i] = U[i] - (dt/dx)*(FR[i] - FL[i]) -(dt/dy)*(FU[i] - FD[i]);
                        U_new[i+NX*NY] = U[i+NX*NY] - (dt/dx)*(FR[i+NX*NY] - FL[i+NX*NY]) -
                                (dt/dy)*(FU[i+NX*NY] - FD[i+NX*NY]);
                        U_new[i+2*NX*NY] = U[i+2*NX*NY] - (dt/dx)*(FR[i+2*NX*NY] - FL[i+2*NX*NY]) -
                                (dt/dy)*(FU[i+2*NX*NY] - FD[i+2*NX*NY]);
                        U_new[i+3*NX*NY] = U[i+3*NX*NY] - (dt/dx)*(FR[i+3*NX*NY] - FL[i+3*NX*NY]) -
                                (dt/dy)*(FU[i+3*NX*NY] - FD[i+3*NX*NY]);

        }

__syncthreads();
        //Renew up and down boundary condition
        if(i >= 1 && i < (NX - 1)) {
                U_new[i] = U_new[i+NX];
                U_new[i+NX*NY] = U_new[i+NX+NX*NY];
                U_new[i+2*NX*NY] = U_new[i+NX+2*NX*NY];
                U_new[i+3*NX*NY] = U_new[i+NX+3*NX*NY];
                U_new[i+(NY-1)*NX] = U_new[i+(NY-2)*NX];
                U_new[i+(NY-1)*NX+NX*NY] = U_new[i+(NY-2)*NX+NX*NY];
                U_new[i+(NY-1)*NX+2*NX*NY] = U_new[i+(NY-2)*NX+2*NX*NY];
                U_new[i+(NY-1)*NX+3*NX*NY] = U_new[i+(NY-2)*NX+3*NX*NY];
        }
 //Renew left and right boundary condition
__syncthreads();
        if(i >= 0 && i < NY){
                U_new[i*NX] = U_new[i*NX+1];
                U_new[i*NX+NX*NY] = U_new[i*NX+1+NX*NY];
                U_new[i*NX+2*NX*NY] = U_new[i*NX+1+2*NX*NY];
                U_new[i*NX+3*NX*NY] = U_new[i*NX+1+3*NX*NY];
                U_new[(NX-1)+i*NX] = U_new[(NX-2)+i*NX];
                U_new[(NX-1)+i*NX+NX*NY] = U_new[(NX-2)+i*NX+NX*NY];
                U_new[(NX-1)+i*NX+2*NX*NY] = U_new[(NX-2)+i*NX+2*NX*NY];
                U_new[(NX-1)+i*NX+3*NX*NY] = U_new[(NX-2)+i*NX+3*NX*NY];
        }

        // Update density, velocity, pressure, and U
__syncthreads();
                if(i<N){
                        dens[i] = U_new[i];
                        xv[i] = U_new[i+NX*NY] / U_new[i];
                        yv[i] = U_new[i+2*NX*NY] / U_new[i];
                        press[i] = (GAMA-1) * (U_new[i+3*NX*NY] - 0.5 * dens[i] * (xv[i]*xv[i] + yv[i]*yv[i]));
			U[i] = U_new[i];
                        U[i+NX*NY] = U_new[i+NX*NY];
                        U[i+2*NX*NY] = U_new[i+2*NX*NY];
                        U[i+3*NX*NY] = U_new[i+3*NX*NY];
                }
__syncthreads();
		
}


void Call_CalculateFlux(){
int threadsPerBlock=256;
int blocksPerGrid=(N+threadsPerBlock-1)/threadsPerBlock;
CalculateFlux<<<blocksPerGrid,threadsPerBlock>>>(d_dens,d_xv,d_yv,d_press,d_E,d_F,d_FR,d_FL,d_FU,d_FD,d_U,d_U_new);
}

	

