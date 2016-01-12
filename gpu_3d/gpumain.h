#include <stdio.h>
#include <malloc.h>
#include <math.h>
#include "sys/time.h"

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
#define NO_STEPS 1

#define R (286.9)           // Gas Constant -> unit: J/(kg*K)
#define GAMA (7.0/5.0)    // Ratio of specific heats
#define CV (R/(GAMA-1.0)) // Cv
#define CP (CV + R)       // Cp

void Allocate_Memory();
void Load_Dat_To_Array(char* input_file_name, float* output_body_array);
void Init();
void Send_To_Device();
void Call_CalculateFlux();
void Get_From_Device();
void Save_Data_To_File(char *output_file_name);
void Free_Memory();
