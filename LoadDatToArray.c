#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#define DEPTH  100								// total grids in x direction
#define WIDTH  100								// total grids in y direction
#define HEIGHT 100								// total grids in z direction

void InitBody(float* body);
void FileToArray(char* filename, float* body);

int main () {
	char *filename = "Export_50x50x50.dat";
	float *body;
	body = (float*)malloc(DEPTH*WIDTH*HEIGHT*sizeof(float));
	// Initialize the body
	InitBody(body);
	// Put cube into the body
	FileToArray(filename, body);
	free(body);
	return 0;
}


void InitBody(float* body) {
	body[DEPTH*WIDTH*HEIGHT] = 0.0;
}


void FileToArray(char* filename, float* body) {
	FILE *pFile;
	pFile = fopen(filename, "r");
	if (!pFile) { printf("Open failure"); } 
    
	char line[60];
     
	int x, y, z, idx;
	float tmp1, tmp2, tmp3;
    
	for(x = 25; x < 75; x++) {
		for(y = 25; y < 75; y++) {
			for(z = 25; z < 75; z++) {
				idx = x*50*50 + y*50 + z;
 
				while (fgets(line, sizeof(line), pFile) != NULL) {	
					fscanf(pFile, "%f%f%f%f", &tmp1, &tmp2, &tmp3, &body[idx]);		    	
					/* test... 0.040018	 -0.204846	 -0.286759	 -1 */		        	
					//printf("%f\n", body[idx]);
					//if(body[idx] == 0) { system("pause"); }
				}		    	    
			}
		}
	}
	fclose(pFile); 
}

