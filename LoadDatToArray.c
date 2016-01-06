#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#define DEPTH  100			// total grids in x direction
#define WIDTH  100			// total grids in y direction
#define HEIGHT 100			// total grids in z direction

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
	int x, y, z, idx;
	for(x = 0; x < DEPTH; x++) {
    		for(y = 0; y < WIDTH; y++) {
    			for(z = 0; z < HEIGHT; z++) {
				idx = x*WIDTH*HEIGHT + y*HEIGHT + z;
				body[idx] = 0.0;
			}
		}
	}
}


void FileToArray(char* filename, float* body) {
	FILE *pFile;
	pFile = fopen(filename, "r");
	if (!pFile) { printf("Open failure"); } 

	char line[60];

	int x, y, z, idx;
    
	for(x = 25; x < 75; x++) {
		for(y = 25; y < 75; y++) {
			for(z = 25; z < 75; z++) {
				idx = x*50*50 + y*50 + z;
    			
				fgets(line, sizeof(line), pFile);
				char *value = strtok(line, " ");
     
				while (value != NULL) {			    	
					body[idx] = atof(value);
					/* test... */
					// 0.040018	 -0.204846	 -0.286759	 -1
					//printf("%f ", body[idx]);
					value = strtok(NULL, " ");
				}
				/* test... */
				//system("pause");    
			}
		}
    	}
	fclose(pFile); 
}
