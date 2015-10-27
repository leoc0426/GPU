#include <stdio.h>
#include <malloc.h>
#include <math.h>
#include <xmmintrin.h>
#include <immintrin.h>
#include <omp.h>
#include <sys/time.h>

#define num_threads 16 //number of threads wanted to use
#define N 200   // Number of cells
#define L 1.0   // Dimensionless length of tube
#define dx (L/N)   // Width of cell
#define dt 0.01*dx  // Size of time step
#define no_steps 20*N  // No. of time steps
float R = 1.0;           // Dimensionless specific gas constant
float GAMA = (7./5.);     // Ratio of specific heats
float CV =R/(GAMA-1.0);  // Cv
float CP =CV + R;       // Cp
float *olddens;     //density
float *newdens;
float *oldxv;       //velocity in x
float *newxv;
float *oldtemp;     //temprature
float *newtemp;
float *oldpress;    //pressure
float *newpress;
float *cx;
float U[N][3];
float U_new[N][3];
float *F;
float *FL;
float *FR;
float FP[N][3];
float FM[N][3];
float dFP[N][3];
float dFM[N][3];

void Allocate_Memory();
void Init();
void Free();
void CalculateFlux();
void CalculateFPFM();
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
	for(i = 0;i < no_steps;i++)
	{
	  CalculateFlux();
	  CalculateFPFM();
	  CalculateResult();
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
	olddens = (float*)malloc(size);
	newdens = (float*)malloc(size);
        oldxv   = (float*)malloc(size);
       	newxv   = (float*)malloc(size);
       	oldtemp = (float*)malloc(size);
       	newtemp = (float*)malloc(size);
        oldpress= (float*)malloc(size);
        newpress= (float*)malloc(size);
       	cx = (float*)malloc(size);
	F = (float*)malloc(3*sizeof(float));
     	FL= (float*)malloc(3*sizeof(float));
        FR= (float*)malloc(3*sizeof(float));
}
void Init()
{
   int i;
omp_set_num_threads(num_threads);
#pragma omp parallel for
   for(i = 0;i < N;i++)
   {
	if(i > 0.5*N)
	{
	//Initialize the right side gas condition
	  olddens[i] = 0.125;
          oldtemp[i] = 0.1;
	  oldxv[i] = 0.0;
	}
	else
	{
        //Initialize the left side gas condition
	  olddens[i] = 1.0;
          oldtemp[i] = 1.0;
          oldxv[i] = 0.0;
	}
	cx[i] = (i - 0.5)* dx;
	U[i][0] = olddens[i];
        U[i][1] = olddens[i]*oldxv[i];
        U[i][2] = olddens[i]*(CV*oldtemp[i] + 0.5*oldxv[i]*oldxv[i]);

   }
}
void CalculateFlux()
{
	int i,j;
	float a,M,M2,F0,F1,F2;
omp_set_num_threads(num_threads);
#pragma omp parallel for private(F0,F1,F2,a,M,M2)
	for(i =0;i < N;i++)
	{
	        F0 = olddens[i]*oldxv[i];
                F1 = olddens[i]*(oldxv[i]*oldxv[i] + R*oldtemp[i]);
                F2 = oldxv[i]*(U[i][2] + olddens[i]*R*oldtemp[i]);
		 a = sqrt(GAMA*R*oldtemp[i]);
		 M = oldxv[i] / a;
		if (M > 1.0)
		  {M = 1.0;}
		else if(M < -1.0)
		  {M = -1.0;}
		 M2 = M*M;

                FP[i][0] = 0.5*(F0*(M + 1.0) + U[i][0]*a*(1-M2));
                FM[i][0] = -0.5*(F0*(M - 1.0) + U[i][0]*a*(1-M2));
		FP[i][1] = 0.5*(F1*(M + 1.0) + U[i][1]*a*(1-M2));
                FM[i][1] = -0.5*(F1*(M - 1.0) + U[i][1]*a*(1-M2));
		FP[i][2] = 0.5*(F2*(M + 1.0) + U[i][2]*a*(1-M2));
                FM[i][2] = -0.5*(F2*(M - 1.0) + U[i][2]*a*(1-M2));


	}
}
void CalculateFPFM()
{
        int i,k;
	float dFP_left,dFP_right,dFM_left,dFM_right;
omp_set_num_threads(num_threads);
#pragma omp parallel for private(dFP_left,dFP_right,dFM_left,dFM_right)
        for(i = 0;i < N;i++)
        {
                dFP[i][0] = 0.0; dFP[i][1] = 0.0; dFP[i][2] = 0.0;
                dFM[i][0] = 0.0; dFM[i][1] = 0.0; dFM[i][2] = 0.0;
		if((i > 0) && (i<N-1))
		{
//omp_set_num_threads(num_threads);
//#pragma omp parallel for 
			for(k = 0;k < 3;k++)
			{
				 dFP_left = FP[i+1][k] - FP[i][k];
                                 dFP_right = FP[i][k] - FP[i-1][k];
				if (dFP_left *dFP_right > 0)
				{
					if(abs(dFP_right) < abs(dFP_left))
					{
					  dFP[i][k] = dFP_right /dx;
					}
					else
                                          dFP[i][k] = dFP_left /dx;
				}
                                dFM_left = FM[i+1][k] - FM[i][k];
                                dFM_right = FM[i][k] - FM[i-1][k];
                                if (dFM_left *dFM_right > 0)
                                {
                                        if(abs(dFM_right) < abs(dFM_left))
                                        {
                                          dFM[i][k] = dFM_right /dx;
                                        }
                                        else
                                          dFM[i][k] = dFM_left /dx;
                                }

			}
		}
        }
}
void CalculateResult()
{

	int i,j;
//omp_set_num_threads(num_threads);
//#pragma omp parallel for
	for(i = 1;i < (N-1);i++)
	{

		for(j = 0;j < 3;j++)
		{
		  FL[j] = (FP[i-1][j] + 0.5*dx*dFP[i-1][j]) + (FM[i][j] - 0.5*dx*dFM[i][j]);
		  FR[j] = (FP[i][j] + 0.5*dx*dFP[i][j]) + (FM[i+1][j] - 0.5*dx*dFM[i+1][j]);
		  U_new[i][j] = U[i][j] - (dt/dx)*(FR[j]-FL[j]);
		}

		olddens[i] = U_new[i][0];
		oldxv[i] = U_new[i][1]/U_new[i][0];
		oldtemp[i] = ((U_new[i][2]/olddens[i]) - 0.5*oldxv[i]*oldxv[i])/CV;
		oldpress[i] = (oldtemp[i]*R)*olddens[i];


	}
omp_set_num_threads(num_threads);
#pragma omp parallel for
	for(i=1;i<N-1;i++)
	{
	U[i][0]=U_new[i][0];
	U[i][1]=U_new[i][1];
	U[i][2]=U_new[i][2];
	}

}
void Free()
{
	free(olddens);
	free(newdens);
        free(oldxv);
        free(newxv);
        free(oldtemp);
        free(newtemp);
        free(oldpress);
        free(newpress);
        free(cx);
        free(F);
        free(FL);
        free(FR);

}

void Save_Results() {
	FILE *pFile;
	int i;
	printf("Saving...");
	pFile = fopen("Results.txt","w");
	for (i = 0; i < N; i++) {
		fprintf(pFile, "%g\t %g\t %g\n", oldtemp[i],oldxv[i],oldpress[i]);
	}
	fclose(pFile);
	printf("Done.\n");
}
