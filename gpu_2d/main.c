#include "gpumain.h"

extern float *dens;
extern float *d_dens;
extern float *xv;
extern float *d_xv;
extern float *yv;
extern float *d_yv;
extern float *press;
extern float *d_press;
extern float *U;
extern float *d_U;
extern float *d_U_new;
extern float *d_E;
extern float *d_F;
extern float *d_FR;
extern float *d_FL;
extern float *d_FU;
extern float *d_FD;
FILE *pFile;
int main(){

Allocate_Memory();
Init();
Send_To_Device();
pFile = fopen("2DResults.txt", "w");
int i;
for(i=0;i< no_steps;i++){
	Call_CalculateFlux();
//	if (i%10 == 0) {
//		printf("\b\b\b\b\b\b\b");
//		printf("%6.2f%%",(float)i/no_steps*100);
//	}
}
Get_From_Device();
Save_Results();
Free_Memory();
fclose(pFile);
return 0;
}

void Save_Results() {
	int i, j;
	for (j = 0; j < NY; j++) {
		for (i = 0; i < NX; i++) {
			fprintf(pFile, "%d %d %f\n", i+1, j+1,dens[i+j*NX]);
		}
	}
}














