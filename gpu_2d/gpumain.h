#include <stdio.h>
#include <malloc.h>
#include <math.h>
#include <sys/time.h>

#define NX 100            // Number of cells in X direction
#define NY 100            // Number of cells in Y direction
#define N  NX*NY          // Number of total cells
#define L 1.0             // Dimensionless length of surface
#define W 1.0             // Dimensionless width of surface
#define dx (L/NX)         // Lenth of cell
#define dy (W/NY)         // Width of cell
#define dt 0.01*0.02      // Size of time step
#define no_steps 4000     // No. of time steps

#define R (1.0)           // Dimensionless specific gas constant
#define GAMA (7.0/5.0)    // Ratio of specific heats
#define CV (R/(GAMA-1.0)) // Cv
#define CP (CV + R)       // Cp

void Init();
void Allocate_Memory();
void Free_Memory();
void Send_To_Device();
void Get_From_Device();
void Save_Results();
void Call_CalculateFlux();

