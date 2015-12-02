#include <stdio.h>
#include <malloc.h>
#include <math.h>
#include <xmmintrin.h>
#include <immintrin.h>
#include <omp.h>
#include <sys/time.h>

#define num_threads 4 //number of threads wanted to use
#define NX 200   // Number of cells in X direction
#define NY 100   // Number of cells in Y direction
#define N  NX*NY // Number of total cells
#define L 1.0   // Dimensionless length of tube
#define W 0.5   // Dimensionless width of tube
#define dx (L/NX)   // Lenth of cell
#define dy (W/NY)   // Width of cell
#define dt 0.01*0.02  // Size of time step
#define no_steps 4000  // No. of time steps

#define R (1.0)         // Dimensionless specific gas constant
#define GAMA (7.0/5.0)     // Ratio of specific heats
#define CV (R/(GAMA-1.0)) // Cv
#define CP (CV + R)       // Cp

float *dens;     //density
float *xv;       //velocity in x
float *temp;     //temprature
float *press;    //pressure
float U[N][3];
float U_new[N][3];
float FL[N][3];
float FR[N][3];
float FU[N][3];
float FD[N][3];
float speed;
void Allocate_Memory();
void Init();
void Free();
void CalculateFlux();
void CalculateResult();
void Save_Results();

int main()
{
  clock_t start, end;

    struct timeval t1, t2;
    double timecost;

    // start timer
    gettimeofday(&t1, NULL);

	int i;
	Allocate_Memory();
	Init();
omp_set_num_threads(num_threads);
#pragma omp parallel private(speed,i)
{
	for(i = 0;i < no_steps;i++)
	{
	  CalculateFlux();
	  CalculateResult();
	}
}
	Save_Results();
	Free();

	 // stop timer
    gettimeofday(&t2, NULL);

    // compute and print time cost in ms
    timecost = (t2.tv_sec - t1.tv_sec) * 1000.0;    // sec to ms
    timecost += (t2.tv_usec - t1.tv_usec) / 1000.0; // us to ms
    printf("Code time cost: %f\n", timecost);
	return 0;
}
void Allocate_Memory()
{
	size_t size = N*sizeof(float);
	dens = (float*)malloc(size);
        xv   = (float*)malloc(size);
       	temp = (float*)malloc(size);
        press= (float*)malloc(size);
}
void Init()
{
   int i,j;

   for(i = 0;i < NX;i++)
   {
	for( j = 0;j < NY;j++)
	{
	  if(i > 0.5*NX)
	  {
	  //Initialize the right side gas condition
	    dens[i+j*NX] = 0.125;
            temp[i+j*NX] = 0.1;
	    xv[i+j*NX] = 0.0;
	  }
	  else
	  {
          //Initialize the left side gas condition
	    dens[i+j*NX] = 1.0;
            temp[i+j*NX] = 1.0;
            xv[i+j*NX] = 0.0;
	  }
	  U[i+j*NX][0] = dens[i+j*NX];
          U[i+j*NX][1] = dens[i+j*NX]*xv[i+j*NX];
          U[i+j*NX][2] = dens[i+j*NX]*(CV*temp[i+j*NX] + 0.5*xv[i+j*NX]*xv[i+j*NX]);
	}
   }
}
void CalculateFlux()
{
	int i,j;

	//omp_set_num_threads(num_threads);
	#pragma omp for collapse(2) private(speed)
	for(i =1;i < (NX-1);i++)
	{
		for( j = 1;j < (NY-1);j++)
		{
		  speed = sqrt(GAMA*R*temp[i+j*NX]);

                  FL[i+j*NX][0] = 0.5*(dens[i+j*NX-1]*xv[i+j*NX-1] + dens[i+j*NX]*xv[i+j*NX] ) - speed*(U[i+j*NX][0] - U[i+j*NX-1][0]);
                  FR[i+j*NX][0] = 0.5*(dens[i+j*NX]*xv[i+j*NX] + dens[i+j*NX+1]*xv[i+j*NX+1] ) - speed*(U[i+j*NX+1][0] - U[i+j*NX][0]);
		  FL[i+j*NX][1] = 0.5*(dens[i+j*NX-1]*(xv[i+j*NX-1]*xv[i+j*NX-1] + R*temp[i+j*NX-1])
				+ dens[i+j*NX]*(xv[i+j*NX]*xv[i+j*NX] + R*temp[i+j*NX]) ) - speed*(U[i+j*NX][1] - U[i+j*NX-1][1]);
                  FR[i+j*NX][1] = 0.5*(dens[i+j*NX]*(xv[i+j*NX]*xv[i+j*NX] + R*temp[i+j*NX])
				+ dens[i+j*NX+1]*(xv[i+j*NX+1]*xv[i+j*NX+1] + R*temp[i+j*NX+1]) ) - speed*(U[i+j*NX+1][1] - U[i+j*NX][1]);
		  FL[i+j*NX][2] = 0.5*(xv[i+j*NX-1]*(U[i+j*NX-1][2] + dens[i+j*NX-1]*R*temp[i+j*NX-1])
				+ xv[i+j*NX]*(U[i+j*NX][2] + dens[i+j*NX]*R*temp[i+j*NX]) ) - speed*(U[i+j*NX][2] - U[i+j*NX-1][2]);
                  FR[i+j*NX][2] = 0.5*(xv[i+j*NX]*(U[i+j*NX][2] + dens[i+j*NX]*R*temp[i+j*NX])
				 + xv[i+j*NX+1]*(U[i+j*NX+1][2] + dens[i+j*NX+1]*R*temp[i+j*NX+1]) ) - speed*(U[i+j*NX+1][2] - U[i+j*NX][2]);
				 

                  FD[i+j*NX][0] = 0.5*(dens[i+(j-1)*NX]*xv[i+(j-1)*NX] + dens[i+j*NX]*xv[i+j*NX] ) - speed*(U[i+j*NX][0] - U[i+(j-1)*NX][0]);
                  FU[i+j*NX][0] = 0.5*(dens[i+j*NX]*xv[i+j*NX] + dens[i+(j+1)*NX]*xv[i+(j+1)*NX] ) - speed*(U[i+(j+1)*NX][0] - U[i+j*NX][0]);
		  FD[i+j*NX][1] = 0.5*(dens[i+(j-1)*NX]*(xv[i+(j-1)*NX]*xv[i+(j-1)*NX] + R*temp[i+(j-1)*NX])
				+ dens[i+j*NX]*(xv[i+j*NX]*xv[i+j*NX] + R*temp[i+j*NX]) ) - speed*(U[i+j*NX][1] - U[i+(j-1)*NX][1]);
                  FU[i+j*NX][1] = 0.5*(dens[i+j*NX]*(xv[i+j*NX]*xv[i+j*NX] + R*temp[i+j*NX])
				+ dens[i+(j+1)*NX]*(xv[i+(j+1)*NX]*xv[i+(j+1)*NX] + R*temp[i+(j+1)*NX]) ) - speed*(U[i+(j+1)*NX][1] - U[i+j*NX][1]);
		  FD[i+j*NX][2] = 0.5*(xv[i+(j-1)*NX]*(U[i+(j-1)*NX][2] + dens[i+(j-1)*NX]*R*temp[i+(j-1)*NX])
				+ xv[i+j*NX]*(U[i+j*NX][2] + dens[i+j*NX]*R*temp[i+j*NX]) ) - speed*(U[i+j*NX][2] - U[i+(j-1)*NX][2]);
                  FU[i+j*NX][2] = 0.5*(xv[i+j*NX]*(U[i+j*NX][2] + dens[i+j*NX]*R*temp[i+j*NX])
				 + xv[i+(j+1)*NX]*(U[i+(j+1)*NX][2] + dens[i+(j+1)*NX]*R*temp[i+(j+1)*NX]) ) - speed*(U[i+(j+1)*NX][2] - U[i+j*NX][2]);
		}
	}
	#pragma omp barrier
}
void CalculateResult()
{

	int i,j;
//omp_set_num_threads(num_threads);
#pragma omp for collapse(2)
	for(i = 1;i < (NX-1);i++)
	{
		for( j = 1;j < (NY-1);j++)
		{
		  U_new[i+j*NX][0] = U[i+j*NX][0] - (dt/dx)*(FR[i+j*NX][0]-FL[i+j*NX][0]) - (dt/dy)*(FU[i+j*NX][0]-FD[i+j*NX][0]);
		  U_new[i+j*NX][1] = U[i+j*NX][1] - (dt/dx)*(FR[i+j*NX][1]-FL[i+j*NX][1]) - (dt/dy)*(FU[i+j*NX][0]-FD[i+j*NX][0]);
		  U_new[i+j*NX][2] = U[i+j*NX][2] - (dt/dx)*(FR[i+j*NX][2]-FL[i+j*NX][2]) - (dt/dy)*(FU[i+j*NX][0]-FD[i+j*NX][0]);
		}
	}
#pragma omp barrier

	//Renew up and down boundary condition
	for(i = 1 ; i < (NX-1);i++)
	{
		U_new[i][0] = U_new[i+NX][0];
		U_new[i][2] = U_new[i+NX][2];
		U_new[i+(NY-1)*NX][0] = U_new[i+(NY-2)*NX][0];
		U_new[i+(NY-1)*NX][2] = U_new[i+(NY-2)*NX][2];
	}
	//Renew left and right boundary condition
	for(i = 0 ; i < NY;i++)
	{
		U_new[i*NX][0] = U_new[i*NX+1][0];
		U_new[i*NX][2] = U_new[i*NX+1][2];
		U_new[(NX-1)+i*NX][0] = U_new[(NX-2)+i*NX][0];
		U_new[(NX-1)+i*NX][2] = U_new[(NX-2)+i*NX][2];
	}
	
//omp_set_num_threads(num_threads);
#pragma omp for collapse(2)
	for(i=0;i < NX;i++)
	{
		for( j = 0;j < NY;j++)
		{
		  dens[i+j*NX]= U_new[i+j*NX][0];
		  xv[i+j*NX]= U_new[i+j*NX][1]/U_new[i+j*NX][0];
		  temp[i+j*NX] = ((U_new[i+j*NX][2]/dens[i+j*NX]) - 0.5*xv[i+j*NX]*xv[i+j*NX])/CV;
		  press[i+j*NX] = (temp[i+j*NX]*R)*dens[i+j*NX];
		  U[i+j*NX][0]=U_new[i+j*NX][0];
		  U[i+j*NX][1]=U_new[i+j*NX][1];
		  U[i+j*NX][2]=U_new[i+j*NX][2];
		}
	}
#pragma omp barrier
}
void Free()
{
	free(dens);
        free(xv);
        free(temp);
        free(press);

}

void Save_Results() {
	FILE *pFile;
	int i,j;
	printf("Saving...");
	pFile = fopen("Results.txt","w");
	for (j = 0; j < NY ;j++) {
	for (i = 0; i < NX; i++) {
		fprintf(pFile, "%d\t %d\t %g\t %g\t %g\n",i+1,j+1,temp[i+j*NX],xv[i+j*NX],press[i+j*NX]);
	}
	}
	fclose(pFile);
	printf("Done.\n");
}
