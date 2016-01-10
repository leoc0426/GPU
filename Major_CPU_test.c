#include <stdio.h>
#include <math.h>

#define NX 100                          // No. of cells in x direction
#define NY 100                          // No. of cells in y direction
#define NZ 100                          // No. of cells in z direction
#define N (NX*NY*NZ)            // N = total number of cells in domain
#define L 100                             // L = length of domain (m)
#define H 100                             // H = Height of domain (m)
#define W 100                             // W = Width of domain (m)
#define DX (L/NX)                       // DX, DY, DZ = grid spacing in x,y,z.
#define DY (H/NY)
#define DZ (W/NZ)
#define DT 0.00001                       // Time step (seconds)
#define NO_STEPS 15

#define R (286.9)           // Gas Constant -> unit: J/(kg*K)
#define GAMMA (7.0/5.0)    // Ratio of specific heats
#define CV (R/(GAMMA-1.0)) // Cv
#define CP (CV + R)       // Cp

//#define DEBUG_PROPERTY
//#define DEBUG_U_for_flux
//#define DEBUG_flux
//#define DEBUG_U_new
//#define DEBUG_U_new_AFTER_REPLACE

#define CPU

void Allocate_Memory();
void Load_Dat_To_Array(char *input_file_name, float *output_body_array);
void Init();

#ifdef CPU
void CPU_Calculate_Flux();
void CalRenewResult();
#endif

void Free_Memory();
void Save_Data_To_File(char *output_file_name);

#ifdef GPU
void Call_GPUHeatContactFunction();
void Call_GPUTimeStepFunction();
void Send_To_Device();
void Get_From_Device();
#endif
float density_a;
float u_a;
float v_a;
float w_a;
float pressure_a;
float temperature_a;

float density_b;
float u_b;
float v_b;
float w_b;
float pressure_b;
float temperature_b;

float density;
float u;
float v;
float w;
float pressure;
float temperature;

//float speed;

float density_for_flux[12];				//density
float u_for_flux[12];					//velocity in x
float v_for_flux[12];					//velocity in y
float w_for_flux[12];					//velocity in z
float pressure_for_flux[12];				//pressure
float temperature_for_flux[12];			//temperature
float speed_for_flux[12];

float U_for_flux[12][5];
float F_for_flux[12][5];

//float *d_dens;              //density 
//float *d_temperature;       //temperature
//float *d_xv;                //velocity in x
//float *d_yv;                //velocity in y
//float *d_zv;                //velocity in z
//float *d_press;             //pressure

float *U;
float *U_new;
//float *E;
//float *F;
//float *G;
float *Rusanov_L;
float *Rusanov_R;
float *Rusanov_F;
float *Rusanov_B;
float *Rusanov_D;
float *Rusanov_U;

float *h_body;
float *d_body;

int total_cells = 0;            // A counter for computed cells

int main() {
	int t;
	// Need to allocate memory first
	Allocate_Memory();

	char *input_file_name = "Export_50x50x50.dat";
	Load_Dat_To_Array(input_file_name, h_body);
	Init();

	for (t = 0; t < NO_STEPS; t++) {
		CPU_Calculate_Flux();
		CalRenewResult();
	}

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

void Allocate_Memory() {
	size_t size = N*sizeof(float);
	h_body = (float*)malloc(size);

	size_t size5 = N*sizeof(float)* 5;
	U = (float*)malloc(size5);
	U_new = (float*)malloc(size5);
	Rusanov_L = (float*)malloc(size5);
	Rusanov_R = (float*)malloc(size5);
	Rusanov_F = (float*)malloc(size5);
	Rusanov_B = (float*)malloc(size5);
	Rusanov_D = (float*)malloc(size5);
	Rusanov_U = (float*)malloc(size5);

#ifdef GPU
	cudaError_t Error;
	Error = cudaMalloc((void**)&d_dens, size);
	printf("CUDA error (malloc d_dens) = %s\n", cudaGetErrorString(Error));
	Error = cudaMalloc((void**)&d_temperature, size);
	printf("CUDA error (malloc d_temperature) = %s\n", cudaGetErrorString(Error));
	Error = cudaMalloc((void**)&d_xv, size);
	printf("CUDA error (malloc d_xv) = %s\n", cudaGetErrorString(Error));
	Error = cudaMalloc((void**)&d_yv, size);
	printf("CUDA error (malloc d_yv) = %s\n", cudaGetErrorString(Error));
	Error = cudaMalloc((void**)&d_zv, size);
	printf("CUDA error (malloc d_zv) = %s\n", cudaGetErrorString(Error));
	Error = cudaMalloc((void**)&d_press, size);
	printf("CUDA error (malloc d_press) = %s\n", cudaGetErrorString(Error));

	/*Error = cudaMalloc((void**)&U, size2);
	printf("CUDA error (malloc U) = %s\n",cudaGetErrorString(Error));
	Error = cudaMalloc((void**)&U_new, size2);
	printf("CUDA error (malloc U_new) = %s\n",cudaGetErrorString(Error));
	Error = cudaMalloc((void**)&E, size2);
	printf("CUDA error (malloc E) = %s\n",cudaGetErrorString(Error));
	Error = cudaMalloc((void**)&F, size2);
	printf("CUDA error (malloc F) = %s\n",cudaGetErrorString(Error));
	Error = cudaMalloc((void**)&G, size2);
	printf("CUDA error (malloc G) = %s\n",cudaGetErrorString(Error));
	Error = cudaMalloc((void**)&FF, size2);
	printf("CUDA error (malloc FF) = %s\n",cudaGetErrorString(Error));
	Error = cudaMalloc((void**)&FB, size2);
	printf("CUDA error (malloc FB) = %s\n",cudaGetErrorString(Error));
	Error = cudaMalloc((void**)&FR, size2);
	printf("CUDA error (malloc FR) = %s\n",cudaGetErrorString(Error));
	Error = cudaMalloc((void**)&FL, size2);
	printf("CUDA error (malloc FL) = %s\n",cudaGetErrorString(Error));
	Error = cudaMalloc((void**)&FU, size2);
	printf("CUDA error (malloc FU) = %s\n",cudaGetErrorString(Error));
	Error = cudaMalloc((void**)&FD, size2);
	printf("CUDA error (malloc FD) = %s\n",cudaGetErrorString(Error));*/

	Error = cudaMalloc((void**)&d_body, size3);
	printf("CUDA error (malloc d_body) = %s\n", cudaGetErrorString(Error));
#endif
}

void Load_Dat_To_Array(char* input_file_name, float* output_body_array) {
	FILE *pFile;
	pFile = fopen(input_file_name, "r");
	if (!pFile) { printf("Open failure\n"); }

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
				fscanf(pFile, "%f%f%f%f", &tmp1, &tmp2, &tmp3, &output_body_array[idx]);
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
					density_b = 1.0;
					u_b = 1.0;
					v_b = 1.0;
					w_b = 1.0;;
					pressure_b = 1.0;
					temperature_b = 1.0;
				}
				else { // air reference: http://www.engineeringtoolbox.com/standard-atmosphere-d_604.html
					density_a = 0.00082 * 0.1;		// unit: kg / m^3
					u_a = -7000;						// unit: m / s
					v_a = 0.0;						// unit: m / s
					w_a = 0.0;						// unit: m / s
					pressure_a = 0.00052 * 10000;		// unit: N / m^2
					temperature_a = -53 + 273.15;		// unit: K
				}
			}
		}
	}

	for (i = 0; i < NX; i++) {
		for (j = 0; j < NY; j++) {
			for (k = 0; k < NZ; k++) {
				if (h_body[i + j*NX + k*NX*NY] == -1.0) { // air
					U[i + j*NX + k*NX*NY + 0 * N] = density_a;
					U[i + j*NX + k*NX*NY + 1 * N] = density_a * u_a;
					U[i + j*NX + k*NX*NY + 2 * N] = density_a * v_a;
					U[i + j*NX + k*NX*NY + 3 * N] = density_a * w_a;
					U[i + j*NX + k*NX*NY + 4 * N] = density_a * (CV*(pressure_a / density_a / R) + 0.5*((u_a*u_a) + (v_a*v_a) + (w_a*w_a)));
				}
				else { // body : h_body[i + j*NX + k*NX*NY] == 0.0
					U[i + j*NX + k*NX*NY + 0 * N] = density_b;
					U[i + j*NX + k*NX*NY + 1 * N] = density_b * u_b;
					U[i + j*NX + k*NX*NY + 2 * N] = density_b * v_b;
					U[i + j*NX + k*NX*NY + 3 * N] = density_b * w_b;
					U[i + j*NX + k*NX*NY + 4 * N] = density_b * (CV*(pressure_b / density_b / R) + 0.5*((u_b*u_b) + (v_b*v_b) + (w_b*w_b)));
				}
			}
		}
	}
}

void CPU_Calculate_Flux() {
	int i, j, k, z, d ,vs;

	//#pragma omp parallel 
	{
		//#pragma omp for
		for (k = 1; k < (NZ - 1); k++) {
			for (j = 1; j < (NY - 1); j++) {
				for (i = 1; i < (NX - 1); i++) {
					for (z = 0; z < 5; z++) {
						//// 問立昕
						vs = ((z % 4) == 0) ? (1) : (-1);
						// no slip condition of U
						// idx back is body
						if (h_body[(i - 1) + j*NX + k*NX*NY] == 0.0) {
							if (h_body[i + j*NX + k*NX*NY] == -1.0) { // idx is air
								U_for_flux[1][z] = U[i + j*NX + k*NX*NY + z*N]; // i air 
								U_for_flux[0][z] = vs*U[i + j*NX + k*NX*NY + z*N]; // B air replace body
							}
							else { // idx is body
								U_for_flux[1][z] = U[i + j*NX + k*NX*NY + z*N]; // i body
								U_for_flux[0][z] = U[(i - 1) + j*NX + k*NX*NY + z*N]; // B body
							}
						}
						else { // idx back is air .
							if (h_body[i + j*NX + k*NX*NY] == -1.0) { // idx is air						
								U_for_flux[1][z] = U[i + j*NX + k*NX*NY + z*N]; // i air
								U_for_flux[0][z] = U[(i - 1) + j*NX + k*NX*NY + z*N]; // B air
							}
							else { // idx is body
								U_for_flux[1][z] = vs*U[(i - 1) + j*NX + k*NX*NY + z*N]; // i air replace body
								U_for_flux[0][z] = U[(i - 1) + j*NX + k*NX*NY + z*N]; // B air 
							}
						}

						// idx front is body
						if (h_body[(i + 1) + j*NX + k*NX*NY] == 0.0) {
							if (h_body[i + j*NX + k*NX*NY] == -1.0) { // idx is air						
								U_for_flux[3][z] = vs*U[i + j*NX + k*NX*NY + z*N]; // F air replace body 
								U_for_flux[2][z] = U[i + j*NX + k*NX*NY + z*N]; // i air 
							}
							else { // idx is body
								U_for_flux[3][z] = U[(i + 1) + j*NX + k*NX*NY + z*N]; // F body 
								U_for_flux[2][z] = U[i + j*NX + k*NX*NY + z*N]; // i body 
							}
						}
						else { // idx front is air 
							if (h_body[i + j*NX + k*NX*NY] == -1.0) { // idx is air						
								U_for_flux[3][z] = U[(i + 1) + j*NX + k*NX*NY + z*N]; // F air 
								U_for_flux[2][z] = U[i + j*NX + k*NX*NY + z*N]; // i air 
							}
							else { // idx is body
								U_for_flux[3][z] = U[(i + 1) + j*NX + k*NX*NY + z*N]; // F air 
								U_for_flux[2][z] = vs*U[(i + 1) + j*NX + k*NX*NY + z*N]; // i air replace body  
							}
						}

						// idx right is body
						if (h_body[i + (j - 1)*NX + k*NX*NY] == 0.0) {
							if (h_body[i + j*NX + k*NX*NY] == -1.0) { // idx is air
								U_for_flux[5][z] = U[i + j*NX + k*NX*NY + z*N]; // i air 
								U_for_flux[4][z] = vs*U[i + j*NX + k*NX*NY + z*N]; // R air replace body
							}
							else { // idx is body
								U_for_flux[5][z] = U[i + j*NX + k*NX*NY + z*N]; // i body
								U_for_flux[4][z] = U[i + (j - 1)*NX + k*NX*NY + z*N]; // R body
							}
						}
						else { // idx right is air 
							if (h_body[i + j*NX + k*NX*NY] == -1.0) { // idx is air						
								U_for_flux[5][z] = U[i + j*NX + k*NX*NY + z*N]; // i air
								U_for_flux[4][z] = U[i + (j - 1)*NX + k*NX*NY + z*N]; // R air
							}
							else { // idx is body
								U_for_flux[5][z] = vs*U[i + (j - 1)*NX + k*NX*NY + z*N]; // i air replace body
								U_for_flux[4][z] = U[i + (j - 1)*NX + k*NX*NY + z*N]; // R air 
							}
						}

						// idx left is body
						if (h_body[i + (j + 1)*NX + k*NX*NY] == 0.0) {
							if (h_body[i + j*NX + k*NX*NY] == -1.0) { // idx is air
								U_for_flux[7][z] = vs*U[i + j*NX + k*NX*NY + z*N]; // L air replace body
								U_for_flux[6][z] = U[i + j*NX + k*NX*NY + z*N]; // i air
							}
							else { // idx is body
								U_for_flux[7][z] = U[i + (j + 1)*NX + k*NX*NY + z*N]; // L body
								U_for_flux[6][z] = U[i + j*NX + k*NX*NY + z*N]; // i body
							}
						}
						else { // idx left is air 
							if (h_body[i + j*NX + k*NX*NY] == -1.0) { // idx is air						
								U_for_flux[7][z] = U[i + (j + 1)*NX + k*NX*NY + z*N]; // L air
								U_for_flux[6][z] = U[i + j*NX + k*NX*NY + z*N]; // i air
							}
							else { // idx is body
								U_for_flux[7][z] = U[i + (j + 1)*NX + k*NX*NY + z*N]; // L air
								U_for_flux[6][z] = vs*U[i + (j + 1)*NX + k*NX*NY + z*N]; // i air replace body
							}
						}

						// idx down is body
						if (h_body[i + j*NX + (k - 1)*NX*NY] == 0.0) {
							if (h_body[i + j*NX + k*NX*NY] == -1.0) { // idx is air
								U_for_flux[9][z] = U[i + j*NX + k*NX*NY + z*N]; // i air 
								U_for_flux[8][z] = vs*U[i + j*NX + k*NX*NY + z*N]; // D air replace body
							}
							else { // idx is body
								U_for_flux[9][z] = U[i + j*NX + k*NX*NY + z*N]; // i body
								U_for_flux[8][z] = U[i + j*NX + (k - 1)*NX*NY + z*N]; // D body
							}
						}
						else { // idx down is air 
							if (h_body[i + j*NX + k*NX*NY] == -1.0) { // idx is air						
								U_for_flux[9][z] = U[i + j*NX + k*NX*NY + z*N]; // i air 
								U_for_flux[8][z] = U[i + j*NX + (k - 1)*NX*NY + z*N]; // D air
							}
							else { // idx is body
								U_for_flux[9][z] = vs*U[i + j*NX + (k - 1)*NX*NY + z*N];	 // i air replace body
								U_for_flux[8][z] = U[i + j*NX + (k - 1)*NX*NY + z*N]; // D air
							}
						}

						// idx up is body
						if (h_body[i + j*NX + (k + 1)*NX*NY] == 0.0) {
							if (h_body[i + j*NX + k*NX*NY] == -1.0) { // idx is air
								U_for_flux[11][z] = vs*U[i + j*NX + k*NX*NY + z*N];	// U air replace body
								U_for_flux[10][z] = U[i + j*NX + k*NX*NY + z*N];	// i body
							}
							else { // idx is body
								U_for_flux[11][z] = U[i + j*NX + (k + 1)*NX*NY + z*N];	// U body
								U_for_flux[10][z] = U[i + j*NX + k*NX*NY + z*N];		// i body
							}
						}
						else { // idx up is air 
							if (h_body[i + j*NX + k*NX*NY] == -1.0) { // idx is air
								U_for_flux[11][z] = U[i + j*NX + (k + 1)*NX*NY + z*N];	// U air
								U_for_flux[10][z] = U[i + j*NX + k*NX*NY + z*N];		// i air
							}
							else { // idx is body
								U_for_flux[11][z] = U[i + j*NX + (k + 1)*NX*NY + z*N];	// U air
								U_for_flux[10][z] = vs*U[i + j*NX + (k + 1)*NX*NY + z*N];	// i air replace body
							}
						}
					}
#ifdef DEBUG_U_for_flux
					/* DDDDDEBUG */
					if (!h_body[(i - 1) + j*NX + k*NX*NY] || !h_body[(i + 1) + j*NX + k*NX*NY] || !h_body[i + (j - 1)*NX + k*NX*NY] || !h_body[i + (j + 1)*NX + k*NX*NY] || !h_body[i + j*NX + (k - 1)*NX*NY] || !h_body[i + j*NX + (k + 1)*NX*NY]) { // idx near body
						printf("h_body[%d + %d*NX + %d*NX*NY] = %f\n", i, j, k, h_body[i + j*NX + k*NX*NY]);
						printf("h_body[(%d-1) + %d*NX + %d*NX*NY] = %f\n", i, j, k, h_body[(i - 1) + j*NX + k*NX*NY]);
						printf("h_body[(%d+1) + %d*NX + %d*NX*NY] = %f\n", i, j, k, h_body[(i + 1) + j*NX + k*NX*NY]);
						printf("h_body[%d + (%d-1)*NX + %d*NX*NY] = %f\n", i, j, k, h_body[i + (j - 1)*NX + k*NX*NY]);
						printf("h_body[%d + (%d+1)*NX + %d*NX*NY] = %f\n", i, j, k, h_body[i + (j + 1)*NX + k*NX*NY]);
						printf("h_body[%d + %d*NX + (%d-1)*NX*NY] = %f\n", i, j, k, h_body[i + j*NX + (k - 1)*NX*NY]);
						printf("h_body[%d + %d*NX + (%d+1)*NX*NY] = %f\n", i, j, k, h_body[i + j*NX + (k + 1)*NX*NY]);
						for (d = 0; d < 12; d++) {
							for (z = 0; z < 5; z++) {
								printf("U_for_flux[%d][%d] = %f\n", d, z, U_for_flux[d][z]);
							}
							printf("\n");
						}
						system("pause");
					}
#endif // DEBUG_U_for_flux
					//// 問立昕
					for (d = 0; d < 12; d++) {
						density_for_flux[d] = U_for_flux[d][0];
						u_for_flux[d] = U_for_flux[d][1] / density_for_flux[d];
						v_for_flux[d] = U_for_flux[d][2] / density_for_flux[d];
						w_for_flux[d] = U_for_flux[d][3] / density_for_flux[d];
						pressure_for_flux[d] = (U_for_flux[d][4] / density_for_flux[d] - 0.5*((u_for_flux[d] * u_for_flux[d]) + (v_for_flux[d] * v_for_flux[d]) + (w_for_flux[d] * w_for_flux[d]))) * density_for_flux[d] * R / CV;
						temperature_for_flux[d] = pressure_for_flux[d] / density_for_flux[d] / R;

						speed_for_flux[d] = sqrt(GAMMA*pressure_for_flux[d] / density_for_flux[d]);		// speed of sound in air

#ifdef DEBUG_PROPERTY
						if (pressure_for_flux[d] > 1000 || pressure_for_flux[d] < 0.0) {
							printf("i = %d, j = %d, k=%d\n", i, j, k);
							printf("U_for_flux[%d][4] = %f\n", d, U_for_flux[d][4]);
							printf("speed[%d] = %f\n", d, speed_for_flux[d]);
							printf("density[%d] = %f\n", d, density_for_flux[d]);
							printf("u[%d] = %f\n", d, u_for_flux[d]);
							printf("v[%d] = %f\n", d, v_for_flux[d]);
							printf("w[%d] = %f\n", d, w_for_flux[d]);
							printf("pressure[%d] = %f\n", d, pressure_for_flux[d]);
							printf("temperature[%d] = %f\n", d, temperature_for_flux[d]);
							system("pause");
						}
#endif

						//// 問立昕
						if ((d == 0) || (d == 1) || (d == 2) || (d == 3)) {
							F_for_flux[d][0] = density_for_flux[d] * u_for_flux[d];
							F_for_flux[d][1] = density_for_flux[d] * u_for_flux[d] * u_for_flux[d] + pressure_for_flux[d];
							F_for_flux[d][2] = density_for_flux[d] * u_for_flux[d] * v_for_flux[d];
							F_for_flux[d][3] = density_for_flux[d] * u_for_flux[d] * w_for_flux[d];
							F_for_flux[d][4] = u_for_flux[d] * (U_for_flux[d][4] + pressure_for_flux[d]);
						}
						else if ((d == 4) || (d == 5) || (d == 6) || (d == 7)) {
							F_for_flux[d][0] = density_for_flux[d] * v_for_flux[d];
							F_for_flux[d][1] = density_for_flux[d] * u_for_flux[d] * v_for_flux[d];
							F_for_flux[d][2] = density_for_flux[d] * v_for_flux[d] * v_for_flux[d] + pressure_for_flux[d];
							F_for_flux[d][3] = density_for_flux[d] * v_for_flux[d] * w_for_flux[d];
							F_for_flux[d][4] = v_for_flux[d] * (U_for_flux[d][4] + pressure_for_flux[d]);
						}
						else if ((d == 8) || (d == 9) || (d == 10) || (d == 11)) {
							F_for_flux[d][0] = density_for_flux[d] * w_for_flux[d];
							F_for_flux[d][1] = density_for_flux[d] * u_for_flux[d] * w_for_flux[d];
							F_for_flux[d][2] = density_for_flux[d] * v_for_flux[d] * w_for_flux[d];
							F_for_flux[d][3] = density_for_flux[d] * w_for_flux[d] * w_for_flux[d] + pressure_for_flux[d];
							F_for_flux[d][4] = w_for_flux[d] * (U_for_flux[d][4] + pressure_for_flux[d]);
						}
					}
#ifdef DEBUG_flux
					/* DDDDDEBUG */
					//if (!h_body[(i - 1) + j*NX + k*NX*NY] || !h_body[(i + 1) + j*NX + k*NX*NY] || !h_body[i + (j - 1)*NX + k*NX*NY] || !h_body[i + (j + 1)*NX + k*NX*NY] || !h_body[i + j*NX + (k - 1)*NX*NY] || !h_body[i + j*NX + (k + 1)*NX*NY]) { // idx near body
					if (!h_body[i + j*NX + k*NX*NY]) { // idx is body
						printf("h_body[%d + %d*NX + %d*NX*NY] = %f\n", i, j, k, h_body[i + j*NX + k*NX*NY]);
						printf("h_body[(%d-1) + %d*NX + %d*NX*NY] = %f\n", i, j, k, h_body[(i - 1) + j*NX + k*NX*NY]);
						printf("h_body[(%d+1) + %d*NX + %d*NX*NY] = %f\n", i, j, k, h_body[(i + 1) + j*NX + k*NX*NY]);
						printf("h_body[%d + (%d-1)*NX + %d*NX*NY] = %f\n", i, j, k, h_body[i + (j - 1)*NX + k*NX*NY]);
						printf("h_body[%d + (%d+1)*NX + %d*NX*NY] = %f\n", i, j, k, h_body[i + (j + 1)*NX + k*NX*NY]);
						printf("h_body[%d + %d*NX + (%d-1)*NX*NY] = %f\n", i, j, k, h_body[i + j*NX + (k - 1)*NX*NY]);
						printf("h_body[%d + %d*NX + (%d+1)*NX*NY] = %f\n", i, j, k, h_body[i + j*NX + (k + 1)*NX*NY]);
						for (d = 0; d < 12; d++) {
							for (z = 0; z < 5; z++) {
								printf("F_for_flux[%d][%d] = %f\n", d, z, F_for_flux[d][z]);
							}
							printf("\n");
						}
						system("pause");
					}
#endif // DEBUG_flux
					for (z = 0; z < 5; z++) {
						//// 問立昕
						// LF flux
						Rusanov_B[i + j*NX + k*NX*NY + z*N] = 0.5*(F_for_flux[0][z] + F_for_flux[1][z] - DX / DT / 3 * (U_for_flux[1][z] - U_for_flux[0][z]));
						Rusanov_F[i + j*NX + k*NX*NY + z*N] = 0.5*(F_for_flux[2][z] + F_for_flux[3][z] - DX / DT / 3 * (U_for_flux[3][z] - U_for_flux[2][z]));

						Rusanov_R[i + j*NX + k*NX*NY + z*N] = 0.5*(F_for_flux[4][z] + F_for_flux[5][z] - DX / DT / 3 * (U_for_flux[5][z] - U_for_flux[4][z]));
						Rusanov_L[i + j*NX + k*NX*NY + z*N] = 0.5*(F_for_flux[6][z] + F_for_flux[7][z] - DX / DT / 3 * (U_for_flux[7][z] - U_for_flux[6][z]));

						Rusanov_D[i + j*NX + k*NX*NY + z*N] = 0.5*(F_for_flux[8][z] + F_for_flux[9][z] - DX / DT / 3 * (U_for_flux[9][z] - U_for_flux[8][z]));
						Rusanov_U[i + j*NX + k*NX*NY + z*N] = 0.5*(F_for_flux[10][z] + F_for_flux[11][z] - DX / DT / 3 * (U_for_flux[11][z] - U_for_flux[10][z]));
					}
				}
			}
		}
		//#pragma omp barrier
	}
}

void CalRenewResult() {
	int i, j, k, z;
	//#pragma omp parallel 
	{
		// Update U by FVM
		//#pragma omp for //collapse(2)
		for (k = 1; k < (NZ - 1); k++) {
			for (j = 1; j < (NY - 1); j++) {
				for (i = 1; i < (NX - 1); i++) {
					if (h_body[i + j*NX + k*NX*NY] == -1.0) { // air
						for (z = 0; z < 5; z++) {
							U_new[i + j*NX + k*NX*NY + z*N] = U[i + j*NX + k*NX*NY + z*N]
								- (DT / DX)*(Rusanov_F[i + j*NX + k*NX*NY + z*N] - Rusanov_B[i + j*NX + k*NX*NY + z*N])
								- (DT / DY)*(Rusanov_L[i + j*NX + k*NX*NY + z*N] - Rusanov_R[i + j*NX + k*NX*NY + z*N])
								- (DT / DZ)*(Rusanov_U[i + j*NX + k*NX*NY + z*N] - Rusanov_D[i + j*NX + k*NX*NY + z*N]);

#ifdef DEBUG_U_new
							/* ...test... */
							if (h_body[i + j*NX + k*NX*NY] == -1.0) { // air
								printf("U_new[%d + %d*NX + %d*NX*NY + %d*N] = %f\n\n", i, j, k, z, U_new[i + j*NX + k*NX*NY + z*N]);
								printf("Rusanov_F[%d + %d*NX + %d*NX*NY + %d*N] = %f\n", i, j, k, z, Rusanov_F[i + j*NX + k*NX*NY + z*N]);
								printf("Rusanov_B[%d + %d*NX + %d*NX*NY + %d*N] = %f\n", i, j, k, z, Rusanov_B[i + j*NX + k*NX*NY + z*N]);
								printf("Rusanov_L[%d + %d*NX + %d*NX*NY + %d*N] = %f\n", i, j, k, z, Rusanov_L[i + j*NX + k*NX*NY + z*N]);
								printf("Rusanov_R[%d + %d*NX + %d*NX*NY + %d*N] = %f\n", i, j, k, z, Rusanov_R[i + j*NX + k*NX*NY + z*N]);
								printf("Rusanov_U[%d + %d*NX + %d*NX*NY + %d*N] = %f\n", i, j, k, z, Rusanov_U[i + j*NX + k*NX*NY + z*N]);
								printf("Rusanov_D[%d + %d*NX + %d*NX*NY + %d*N] = %f\n", i, j, k, z, Rusanov_D[i + j*NX + k*NX*NY + z*N]);
								//printf("U_new[(%d-1) + %d*NX + %d*NX*NY + %d*N] = %f\n", i, j, k, z, U_new[(i - 1) + j*NX + k*NX*NY + z*N]);
								//printf("U_new[(%d+1) + %d*NX + %d*NX*NY + %d*N] = %f\n", i, j, k, z, U_new[(i + 1) + j*NX + k*NX*NY + z*N]);
								//printf("U_new[%d + (%d-1)*NX + %d*NX*NY + %d*N] = %f\n", i, j, k, z, U_new[i + (j - 1)*NX + k*NX*NY + z*N]);
								//printf("U_new[%d + (%d+1)*NX + %d*NX*NY + %d*N] = %f\n", i, j, k, z, U_new[i + (j + 1)*NX + k*NX*NY + z*N]);
								//printf("U_new[%d + %d*NX + (%d-1)*NX*NY + %d*N] = %f\n", i, j, k, z, U_new[i + j*NX + (k - 1)*NX*NY + z*N]);
								//printf("U_new[%d + %d*NX + (%d+1)*NX*NY + %d*N] = %f\n", i, j, k, z, U_new[i + j*NX + (k + 1)*NX*NY + z*N]);
								system("pause");
							}
#endif // DEBUG_U_new

						}
					}
				}
			}
		}
		//#pragma omp barrier

		//Renew front boundary condition: fixed condition
		//#pragma omp for
		for (k = 0; k < NZ; k++) {
			for (j = 0; j < NY; j++) {
				for (z = 0; z < 5; z++) {
					// front = front: fixed condition -> 99 = 99
					U_new[(NX - 1) + j*NX + k*NX*NY + z*N] = U[(NX - 1) + j*NX + k*NX*NY + z*N];
				}
			}
		}
		//#pragma omp barrier

		//Renew back boundary condition: free condition
		//#pragma omp for
		for (k = 1; k < (NZ - 1); k++) {
			for (j = 1; j < (NY - 1); j++) {
				for (z = 0; z < 5; z++) {
					// back = back+1 -> 10100 = 10101 
					U_new[0 + j*NX + k*NX*NY + z*N] = U_new[1 + j*NX + k*NX*NY + z*N];
				}
			}
		}
		//#pragma omp barrier

		//Renew right and left boundary condition: free condition
		//#pragma omp for
		for (k = 1; k < (NZ - 1); k++) {
			for (i = 1; i < (NX - 1); i++) {
				for (z = 0; z < 5; z++) {
					// right = right+100 -> 10001 = 10101
					U_new[i + 0 * NX + k*NX*NY + z*N] = U_new[i + 1 * NX + k*NX*NY + z*N];
					// left = left-100 -> 19901 = 19801
					U_new[i + (NY - 1)*NX + k*NX*NY + z*N] = U_new[i + (NY - 2)*NX + k*NX*NY + z*N];
				}
			}
		}
		//#pragma omp barrier

		//Renew down and up boundary condition: free condition
		//#pragma omp for
		for (j = 1; j < (NY - 1); j++) {
			for (i = 1; i < (NX - 1); i++) {
				for (z = 0; z < 5; z++) {
					// down = down+10000 -> 101 = 10100
					U_new[i + j*NX + z*N] = U_new[i + j*NX + NX*NY + z*N];
					// up = up-10000 -> 990101 = 980101
					U_new[i + j*NX + (NZ - 1)*NX*NY + z*N] = U_new[i + j*NX + (NZ - 2)*NX*NY + z*N];
				}
			}
		}
		//#pragma omp barrier

		// 8 edges
		for (z = 0; z < 5; z++) {
			for (i = 0; i < (NX - 1); i++) {
				// 0, 1~98
				U_new[i + z*N] = U[i + z*N];
				// 9900, 9901~9998
				U_new[i + (NY - 1)*NX + z*N] = U[i + (NY - 1)*NX + z*N];
				// 990000, 990001~990098
				U_new[i + (NZ - 1)*NX*NY + z*N] = U[i + (NZ - 1)*NX*NY + z*N];
				// 999900, 999901~999998
				U_new[i + (NY - 1)*NX + (NZ - 1)*NX*NY + z*N] = U[i + (NY - 1)*NX + (NZ - 1)*NX*NY + z*N];
			}
			for (j = 1; j < (NY - 1); j++) {
				// 100~9800
				U_new[j*NY + z*N] = U[j*NY + z*N];
				// 990100~9909800
				U_new[j*NY + (NZ - 1)*NX*NY + z*N] = U[j*NY + (NZ - 1)*NX*NY + z*N];
			}
			for (k = 1; k < (NZ - 1); k++) {
				// 10000~980000
				U_new[k*NY*NX + z*N] = U[k*NY*NX + z*N];
				// 19900~989900
				U_new[k*NY*NX + (NY - 1)*NX + z*N] = U[k*NY*NX + (NY - 1)*NX + z*N];
			}
		}

		// Update density, velocity, pressure, and U
		//#pragma omp for //collapse(2)
		for (k = 0; k < NZ; k++) {
			for (j = 0; j < NY; j++) {
				for (i = 0; i < NX; i++) {
					if (h_body[i + j*NX + k*NX*NY] == -1.0) { // air
						for (z = 0; z < 5; z++) {
							U[i + j*NX + k*NX*NY + z*N] = U_new[i + j*NX + k*NX*NY + z*N];

#ifdef DEBUG_U_new_AFTER_REPLACE
							/* ...test... */
							printf("U_new[%d + %d*NX + %d*NX*NY + %d*N] = %f\n", i, j, k, z, U_new[i + j*NX + k*NX*NY + z*N]);
							system("pause");
#endif // DEBUG_U_new_AFTER_REPLACE

						}
					}
				}
			}
		}
		//#pragma omp barrier
	}
}

#ifdef GPU
void Call_GPUHeatContactFunction() {
	int threadsPerBlock = 256;
	int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
	size_t size = N*sizeof(float);
	cudaError_t Error;
	GPUHeatContactFunction << <blocksPerGrid, threadsPerBlock >> >(d_a, d_b, d_body);
	Error = cudaMemcpy(d_a, d_b, size, cudaMemcpyDeviceToDevice);
	printf("CUDA error (memcpy d_b -> d_a) = %s\n", cudaGetErrorString(Error));
}

__global__ void GPUHeatContactFunction(float *a, float *b, int *body) {

}

void Call_GPUTimeStepFunction() {
	int threadsPerBlock = 256;
	int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
	size_t size = N*sizeof(float);
	cudaError_t Error;
	GPUTimeStepFunction << <blocksPerGrid, threadsPerBlock >> >(d_a, d_b, d_body);
	Error = cudaMemcpy(d_a, d_b, size, cudaMemcpyDeviceToDevice);
	printf("CUDA error (memcpy d_b -> d_a) = %s\n", cudaGetErrorString(Error));
}

__global__ void GPUTimeStepFunction(float *a, float *b, int *body){

}

void Send_To_Device() {
	size_t size = N*sizeof(float);
	cudaError_t Error;
	Error = cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
	printf("CUDA error (memcpy h_a -> d_a) = %s\n", cudaGetErrorString(Error));
	size = N*sizeof(float);
	Error = cudaMemcpy(d_body, h_body, size, cudaMemcpyHostToDevice);
	printf("CUDA error (memcpy h_body -> d_body) = %s\n", cudaGetErrorString(Error));
}

void Get_From_Device() {
	size_t size = N*sizeof(float);
	cudaError_t Error;
	Error = cudaMemcpy(h_a, d_a, size, cudaMemcpyDeviceToHost);
	printf("CUDA error (memcpy d_a -> h_a) = %s\n", cudaGetErrorString(Error));
}
#endif

void Free_Memory() {
	if (h_body) free(h_body);

	if (U) free(U);
	if (U_new) free(U_new);
	if (Rusanov_L) free(Rusanov_L);
	if (Rusanov_R) free(Rusanov_R);
	if (Rusanov_F) free(Rusanov_F);
	if (Rusanov_B) free(Rusanov_B);
	if (Rusanov_D) free(Rusanov_D);
	if (Rusanov_U) free(Rusanov_U);
#ifdef GPU
	if (d_dens) cudaFree(d_dens);
	if (d_temperature) cudafree(d_temperature);
	if (d_xv) cudafree(d_xv);
	if (d_yv) cudafree(d_yv);
	if (d_zv) cudafree(d_zv);
	if (d_press) cudafree(d_press);
	if (d_body) cudaFree(d_body);
#endif
}

void Save_Data_To_File(char *output_file_name) {
	FILE *pOutPutFile;
	pOutPutFile = fopen(output_file_name, "w");
	if (!pOutPutFile) { printf("Open failure"); }

	fprintf(pOutPutFile, "TITLE=\"Flow Field of X-37\"\n");
	fprintf(pOutPutFile, "VARIABLES=\"X\", \"Y\", \"Z\", \"Vx\", \"Vy\", \"Vz\", \"Pressure\", \"Temperature\", \"Body\"\n");
	fprintf(pOutPutFile, "ZONE I = 100, J = 100, K = 100, F = POINT\n");

	int i, j, k;
	for (k = 0; k < NZ; k++) {
		for (j = 0; j < NY; j++) {
			for (i = 0; i < NX; i++) {
				density = U[i + j*NX + k*NX*NY + 0*N];
				u = U[i + j*NX + k*NX*NY + 1*N] / density;
				v = U[i + j*NX + k*NX*NY + 2*N] / density;
				w = U[i + j*NX + k*NX*NY + 3*N] / density;
				pressure = (U[i + j*NX + k*NX*NY + 4 * N] / density - 0.5*((u*u) + (v*v) + (w*w))) *density*R / CV;
				temperature = pressure / density / R;
				fprintf(pOutPutFile, "%d %d %d %f %f %f %f %f %2.0f\n", i, j, k, u, v, w, pressure, temperature, -h_body[i + j*NX + k*NX*NY]);
			}
		}
	}
}

