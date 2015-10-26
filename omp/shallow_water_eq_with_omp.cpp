#include <stdio.h>
#include <math.h>
#include <omp.h>

#define N 200
#define L 1.0
#define DX (L/N)
#define DT (0.1*DX)
#define NO_STEPS 200
#define G 9.81


float U[2][N];    	// Create U as a 2xN array here, U = n and nu
float P[2][N];		// Primitives, n and u
float U_new[2][N];	// New U array

void Save_Results();


int main(int argc, char** argv) {

	int tid;
	int i, j;
	float interface_u;
	float FR[2];
	float FL[2];
	float speed;
	int index;
	int N_thread = N/2; 	// share work
	printf("1D SW Eqn Solver\n");

	// Set the initial condition
	for (i = 0; i < N; i++) {
		if (i < 0.5*N) {
			P[0][i] = 1.0; // Water Depth
			P[1][i] = 0.0; // Water Speed
		} else {
			P[0][i] = 0.1;
			P[1][i] = 0.0;
		}
		// Compute U vector 
		U[0][i] = P[0][i];              // Depth = mass of fluid
		U[1][i] = P[0][i]*P[1][i];      // Momentum of fluid
	}

	omp_set_num_threads(2); // Create 2 threads for this
	#pragma omp parallel private(tid, i, j, FL, FR, speed,index) shared(U, P, U_new, N_thread)
	{

	tid = omp_get_thread_num();
	printf("Thread %d up and running\n", tid);
	
	for (j = 0; j < NO_STEPS; j++) {


		// Compute U_new in all cells (except the ends)
		for (index = 0; index < N_thread; index++) {
			i = tid*N_thread + index;
			if ((i > 0) && (i < (N-1))) {
				// Left Flux first - the flux across the surface between i-1 and i 

				// Rusanov Flux
				speed = sqrtf(0.5*G*(P[0][i-1]+P[0][i]));
				FL[0] = 0.5*(P[0][i-1]*P[1][i-1] + P[0][i]*P[1][i]) - speed*(U[0][i] - U[0][i-1]);
				FL[1] = 0.5*(  (P[0][i-1]*P[1][i-1]*P[1][i-1] + 0.5*G*P[0][i-1]*P[0][i-1]) + (P[0][i]*P[1][i]*P[1][i] + 0.5*G*P[0][i]*P[0][i]) ) - speed*(U[1][i] - U[1][i-1]);
			

				// Right Flux next - the flux across the surface between i and i+1
	
				// Rusanov Flux
				speed = sqrtf(0.5*G*(P[0][i+1]+P[0][i]));
				FR[0] = 0.5*(P[0][i]*P[1][i] + P[0][i+1]*P[1][i+1]) - speed*(U[0][i+1] - U[0][i]);
				FR[1] = 0.5*(  (P[0][i]*P[1][i]*P[1][i] + 0.5*G*P[0][i]*P[0][i]) + (P[0][i+1]*P[1][i+1]*P[1][i+1] + 0.5*G*P[0][i+1]*P[0][i+1]) ) - speed*(U[1][i+1] - U[1][i]);
				
	
				// Now, compute the new U value
				U_new[0][i] = U[0][i] - (DT/DX)*(FR[0]-FL[0]);
				U_new[1][i] = U[1][i] - (DT/DX)*(FR[1]-FL[1]);
			}
			
			// We cannot update P, yet. Next loop.
		}
		#pragma omp barrier
		
		// Update U and P now
		for (index = 0; index < N_thread; index++) {
			i = tid*N_thread + index;
			if ( (i > 0) && (i < (N-1)) ) {
				U[0][i] = U_new[0][i];
				U[1][i] = U_new[1][i];
		
				P[0][i] = U[0][i];	
				P[1][i] = U[1][i]/U[0][i];	
			}
		}
		#pragma omp barrier
		
		if (tid == 0) {
			// Correct ends using reflective conditions
			P[0][0] = P[0][1];  
			P[1][0] = -P[1][1]; 
			
			P[0][N-1] = P[0][N-2];  
			P[1][N-1] = -P[1][N-2]; 
		}
		#pragma omp barrier
		
	}
	} // end parallel section
	// Save the data
	Save_Results();	
	
	
	return 0;
}

void Save_Results() {
	FILE *pFile;
	int i;
	printf("Saving...");
	pFile = fopen("Results.txt","w");
	for (i = 0; i < N; i++) {
		fprintf(pFile, "%g\t %g\n", U[0][i], U[1][i]);
	}
	fclose(pFile);	
	printf("Done.\n");
}
