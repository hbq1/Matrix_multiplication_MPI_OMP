#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <time.h>

#define EPS (1e-10)
#define SEED_A (1)
#define SEED_B (2)

int N;

void generate_matrix(float* M, int size, int seed);
void mult_square_matrices(float* A, float* B, float* C, int size);
void print_matrix(float* M, int size);
void transpose(float* M, int size);

void mult_square_matrices_opt(float* A, float* B, float* C, int size);

int main(int argc, char** argv)
{
	N = atoi(argv[1]);
	float *A,  *B, *C;
	double start_time, finish_time;
	double exec_time;
	
	// Allocate memory for all matrices
	A = (float*) calloc(N*N, sizeof(float));
	B = (float*) calloc(N*N, sizeof(float));
	C = (float*) calloc(N*N, sizeof(float));

	// Generate matrices
	generate_matrix(A, N, SEED_A);
	generate_matrix(B, N, SEED_B);
	
	// Multiply matrices
	start_time = omp_get_wtime();;
	mult_square_matrices_opt(A, B, C, N);
	finish_time = omp_get_wtime();

	// Get execution time
	exec_time = (finish_time - start_time);
 
	// Print execution time
	printf("Multiplication time: %.5lf\n", exec_time);

	// Print result
	//print_matrix(A, N);
	//print_matrix(B, N);
	//print_matrix(C, N);

	free(A); free(B); free(C);

	return 0;
}

void mult_square_matrices(float* A, float* B, float* C, int size) 
{
	int i, j, k;
	float buf;
	
	for(i = 0; i < size; ++i)
		for(j = 0; j < size; ++j) {
			C[i*size + j] = 0.0;
			for(k = 0; k < size; ++k)
				C[i*size + j] += A[i*size + k] * B[k*size + j];
		}
}

void mult_square_matrices_opt(float* A, float* B, float* C, int size) 
{
	int i, j, k;
	float buf;
	float *bufA, *bufB, *bufC;
	transpose(B, size);

	for(i = 0; i < size; ++i) {
		bufC = &(C[i*size]);
		bufA = &(A[i*size]);
		for(j = 0; j < size; ++j) {
			bufB = &(B[j*size]);
			bufC[j] = 0.0;

			for(k = 0; k < size; ++k)
				bufC[j] += bufA[k] * bufB[k];
		}
	}
	transpose(B, size);
}


void transpose(float* M, int size)
{
	int i, j;
	float temp;

	for(i = 0; i < size; ++i)
		for(j = 0; j < i; ++j) {
			temp = M[i * size + j];
			M[i * size + j] = M[j * size + i];
			M[j * size + i] = temp;
		}
}

void generate_matrix(float* M, int size, int seed) 
{
	int i, j;
	
	for(i = 0; i < size; ++i)
		for(j = 0; j < size; ++j)
			M[i*size + j] = ((float)(i + j + seed)) / (i + j + 1) * ( (i ^ seed) + 0.55);
}

void print_matrix(float* M, int size)
{
	int i, j;

	printf("(\n");
	for(i = 0; i < size; ++i) {
		for(j = 0; j < size; ++j)
			printf("%.0f\t", M[i*size + j]);
		printf("\n");
	} 
	printf(")\n");
}

