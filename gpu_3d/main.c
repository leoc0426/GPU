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

int main(){
	int t;
	// Need to allocate memory first
	Allocate_Memory();

	char *input_file_name = "Export_50x50x50.dat";
	Load_Dat_To_Array(input_file_name, h_body);
	Init();
	for (t = 0; t < NO_STEPS; t++) {
		
		printf("%3.0f%", (float)t / NO_STEPS);
		printf("\b\b\b");
	}
	printf("100%\n");
	char *output_file_name = "3DResults.dat";
	Save_Data_To_File(output_file_name);

	//// GPU code
	// Send to device
	//Send_To_Device();
	// Call our function
	//Call_GPU_Function();
	// Get from device
	//Get_From_Device();
	// Print out the first 5 values of b

	// Free the memory
	Free_Memory();

return 0;
}
















