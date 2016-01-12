#include "gpumain.h"

extern float *dens;
extern float *d_dens;
extern float *xv;
extern float *d_xv;
extern float *yv;
extern float *d_yv;
extern float *zv;
extern float *d_zv;
extern float *press;
extern float *d_press;
extern float *temperature;
extern float *U;
extern float *d_U;
extern float *d_U_new;
extern float *d_E;
extern float *d_F;
extern float *d_G;
extern float *d_FR;
extern float *d_FL;
extern float *d_FU;
extern float *d_FD;
extern float *d_FB;
extern float *d_FF;
extern float *h_body;

int main() {
	int t;
	// Need to allocate memory first
	Allocate_Memory();

	char *input_file_name = "Export_50x50x50.dat";
	Load_Dat_To_Array(input_file_name, h_body);

	Init();

	struct timeval start, end;
	float time;
	/* start the timer */
	gettimeofday(&start, NULL);
	// Send to device
	Send_To_Device();
	for (t = 0; t < NO_STEPS; t++) {
		// Call our function
		Call_CalculateFlux();
		printf("%3.0f%", (float)t / NO_STEPS);
		printf("\b\b\b");
	}
	printf("100%\n");
	// Get from device
	Get_From_Device();
	gettimeofday(&end, NULL);
	time = ((end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1000000.0);
	printf("Computation Time = %f sec. \n", time);
	/* end the timer */

	char *output_file_name = "3DResults.dat";
	Save_Data_To_File(output_file_name);

	// Free the memory
	Free_Memory();

	return 0;
}
