#include <stdio.h>
#include <malloc.h>
#include <math.h>
#include <xmmintrin.h>
#include <immintrin.h>
#include <omp.h>
#include <sys/time.h>

#define num_threads 16 		//number of threads wanted to use
#define N 200   			// Number of cells
#define L 1.0   			// Dimensionless length of tube
#define dx (L/N)   			// Width of cell
#define dt 0.1*dx  		// Size of time step
#define no_steps 30000  	// No. of time steps
#define R 1.0           	// Dimensionless specific gas constant
#define GAMA (7.0/5.0)     	// Ratio of specific heats
#define CV (R/(GAMA-1.0))  	// Cv
#define CP (CV + R)       	// Cp

float *dens;     			//density
float *xv;       			//velocity in x
float *temp;     			//temprature
float *press;    			//pressure
float *cx;
float U[N][3];
float U_new[N][3];
float *F;
float FL[N][3];
float FR[N][3];

void Allocate_Memory();
void Init();
void Free();
void CalculateFlux();
void CalculateResult();
void Save_Results();

int main() {
  clock_t start, end;
 
    struct timeval t1, t2;
    double timecost;

    // start timer
    gettimeofday(&t1, NULL);

	int i, j;
	Allocate_Memory();
	Init();
	
	FILE *pFile;
	pFile = fopen("Sodtube_Rusanov_Ryan_Results.txt","w");
	
	for(i = 0;i < no_steps;i++)	{
		CalculateFlux();
		CalculateResult();
		if (i%10 == 0) {		
			for (j = 0; j < N; j++) {
				fprintf(pFile, "%g\t %g\t %g\n", temp[j],xv[j],press[j]);
			}
		}		
	}
	fclose(pFile);
	//Save_Results();
	Free();

	// stop timer
    gettimeofday(&t2, NULL);

    // compute and print time cost in ms
    timecost = (t2.tv_sec - t1.tv_sec) * 1000.0;    // sec to ms
    timecost += (t2.tv_usec - t1.tv_usec) / 1000.0; // us to ms
    printf("Code time cost: %f\n", timecost);
	return 0;
}

void Allocate_Memory() {
	size_t size = N*sizeof(float);
	dens = (float*)malloc(size);
    xv   = (float*)malloc(size);
    temp = (float*)malloc(size);
    press= (float*)malloc(size);
    cx = (float*)malloc(size);
	F = (float*)malloc(3*sizeof(float));
}

void Init() {
   int i;
	omp_set_num_threads(num_threads);
	#pragma omp parallel for
   for(i = 0;i < N;i++)	{
		if(i > 0.5*N) {
			//Initialize the right side gas condition
			dens[i] = 0.125;
    		temp[i] = 1.0;
			xv[i] = 0.0;
		} else {
	        //Initialize the left side gas condition
			dens[i] = 1.0;
	        temp[i] = 1.0;
	        xv[i] = 0.0;
		}
		cx[i] = (i - 0.5)* dx;
		U[i][0] = dens[i];
	    U[i][1] = dens[i]*xv[i];
	    U[i][2] = dens[i]*(CV*temp[i] + 0.5*xv[i]*xv[i]);
	}
}

void CalculateFlux() {
	int i,j;
	float speed;
	omp_set_num_threads(num_threads);
	#pragma omp parallel for private(speed)
	for(i =1;i < (N-1);i++)	{
		speed = sqrt(GAMA*R*temp[i]);
        FL[i][0] = 0.5*(dens[i-1]*xv[i-1] + dens[i]*xv[i] ) - speed*(U[i][0] - U[i-1][0]);
        FR[i][0] = 0.5*(dens[i]*xv[i] + dens[i+1]*xv[i+1] ) - speed*(U[i+1][0] - U[i][0]);
        
		FL[i][1] = 0.5*(dens[i-1]*(xv[i-1]*xv[i-1] + R*temp[i-1]) + dens[i]*(xv[i]*xv[i] + R*temp[i]) ) - speed*(U[i][1] - U[i-1][1]);
        FR[i][1] = 0.5*(dens[i]*(xv[i]*xv[i] + R*temp[i]) + dens[i+1]*(xv[i+1]*xv[i+1] + R*temp[i+1]) ) - speed*(U[i+1][1] - U[i][1]);
        
		FL[i][2] = 0.5*(xv[i-1]*(U[i-1][2] + dens[i-1]*R*temp[i-1]) + xv[i]*(U[i][2] + dens[i]*R*temp[i]) ) - speed*(U[i][2] - U[i-1][2]);
        FR[i][2] = 0.5*(xv[i]*(U[i][2] + dens[i]*R*temp[i]) + xv[i+1]*(U[i+1][2] + dens[i+1]*R*temp[i+1]) ) - speed*(U[i+1][2] - U[i][2]);
	}
}

void CalculateResult() {
	int i,j;
	//omp_set_num_threads(num_threads);
	//#pragma omp parallel for
	for(i = 1;i < (N-1);i++) {
		for(j = 0;j < 3;j++) {
			U_new[i][j] = U[i][j] - (dt/dx)*(FR[i][j]-FL[i][j]);
		}
		dens[i] = U_new[i][0];
		xv[i] = U_new[i][1]/U_new[i][0];
		temp[i] = ((U_new[i][2]/dens[i]) - 0.5*xv[i]*xv[i])/CV;
		press[i] = (temp[i]*R)*dens[i];
	}
	omp_set_num_threads(num_threads);
	#pragma omp parallel for
	for(i=1;i<N-1;i++) {
		U[i][0] = U_new[i][0];
		U[i][1] = U_new[i][1];
		U[i][2] = U_new[i][2];
	}
	
	// Boudary condition
	dens[0] = dens[1];
	xv[0] = -xv[1];
	temp[0] = temp[1];
	press[0] = press[1];
	
	dens[N-1] = dens[N-2];
	xv[N-1] = -xv[N-2];
	temp[N-1] = temp[N-2];
	press[N-1] = press[N-2];
		
	U[0][0] = U[1][0];
	U[0][1] = -U[1][1];
	U[0][2] = U[1][2];
	
	U[N-1][0] = U[N-2][0];
	U[N-1][1] = -U[N-2][1];
	U[N-1][2] = U[N-2][2];	
}

void Free() {
	free(dens);
    free(xv);
    free(temp);
    free(press);
    free(cx);
    free(F);
}

void Save_Results() {
	FILE *pFile;
	int i;
	printf("Saving...");
	pFile = fopen("Sodtube_Rusanov_Ryan_Results.txt","w");
	for (i = 0; i < N; i++) {
		fprintf(pFile, "%g\t %g\t %g\n", temp[i],xv[i],press[i]);
	}
	fclose(pFile);
	printf("Done.\n");
}
