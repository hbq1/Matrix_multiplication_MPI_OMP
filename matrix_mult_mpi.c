#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <mpi.h>
#include <time.h>
 	
#define EPS (1e-10)
#define SEED_A (1)
#define SEED_B (2)

int N;

void generate_matrix(float* M, int size, int seed);
void mult_square_matrices(float* A, float* B, float* C, int size);
void print_matrix(float* M, int size);
void transpose(float* M, int size);

int main(int argc, char** argv)
{
	N = atoi(argv[1]);
	float *A,  *B, *C;
	double start_time, finish_time, exec_time;
	double max_exec_time;
	int rank;

	MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	if (rank == 0) {
		// Allocate memory for all matrices
		A = (float*) calloc(N*N, sizeof(float));
		B = (float*) calloc(N*N, sizeof(float));
		C = (float*) calloc(N*N, sizeof(float));

		// Generate matrices
		generate_matrix(A, N, SEED_A);
		generate_matrix(B, N, SEED_B);
	}	
	// Multiply matrices

	start_time = MPI_Wtime();
	mult_square_matrices(A, B, C, N);
	finish_time = MPI_Wtime();

	// Get execution time
	exec_time = (finish_time - start_time);
    	MPI_Reduce(&exec_time, &max_exec_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
	
	if (rank == 0) {
		// Print execution time
		printf("Multiplication time: %.5lf\n", max_exec_time);

		// Print result
		//print_matrix(A, N);
		//print_matrix(B, N);
		//print_matrix(C, N);

		free(A); free(B); free(C);
	}	
	MPI_Finalize();	
	return 0;
}

void mult_square_matrices(float* A, float* B, float* C, int size) 
{
	int i, j, k, p, ind;
	int num_processes, rank;
	int a_part_size, buffer_size;
	int *part_num_elems, *displs;
	float *buf_A, *buf_B, *buf_C;
	//for optimization
	float tmp_accumulator;
	float *opt_A, *opt_B, *opt_C; // for optimization
	
	MPI_Status status;
	
	// get rank & size
     	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
     	MPI_Comm_size(MPI_COMM_WORLD, &num_processes);
	
	// get size for current process
	a_part_size = size / num_processes;
	
	// size with 1 additional line for big parts
	buffer_size = (a_part_size + 1) * size;
	
	buf_A = (float*) calloc(buffer_size, sizeof(float));
	buf_B = (float*) calloc(buffer_size, sizeof(float));
	buf_C = (float*) calloc(buffer_size, sizeof(float));

	// coordinator: preparation
	if (rank == 0) {
		// allocate memory for displacements in result array
		displs = (int*) calloc(num_processes, sizeof(int));
		// determine number of elements on each part
		part_num_elems = (int*) calloc(num_processes, sizeof(int));
		for(i = 0; i < num_processes; ++i) {
			part_num_elems[i] = a_part_size * size;
			// additional size for last lines of matrices
			if (i <  size % num_processes)
				part_num_elems[i] += size;
		}
		// define displacement
		int prev_sum = 0;
		for(i = 0; i < num_processes; ++i) {
			displs[i] = prev_sum;
			prev_sum +=  part_num_elems[i];
		}
		// transpose B for conviniency
		transpose(B, size);
	}

	// scatter last lines of matrix
	if (rank < size % num_processes) {
		a_part_size++;
	}
	
	// scatter matrix lines	
	MPI_Scatterv(A, part_num_elems, displs, MPI_FLOAT, buf_A, a_part_size * size, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Scatterv(B, part_num_elems, displs, MPI_FLOAT, buf_B, a_part_size * size, MPI_FLOAT, 0, MPI_COMM_WORLD);

	// define neightbours
	int next_proc = (rank + 1) % num_processes;
	int prev_proc = (rank - 1 + num_processes) % num_processes;
	int b_part_size = a_part_size;
	
	// initilize index shift for each process
	int ind_shift = 0;
	for(i = 0; i < rank; ++i) 
		ind_shift += ( i < size % num_processes ? size/num_processes + 1 : size/num_processes );

	// move throught all B's stripes ...
	for(p = 0; p < num_processes; ++p) {
		if (p > 0) {
			//shift B stripes
			MPI_Sendrecv_replace(buf_B, buffer_size, MPI_FLOAT, next_proc, 0, prev_proc, 0, MPI_COMM_WORLD, &status);
			MPI_Sendrecv_replace(&b_part_size, 1, MPI_INT, next_proc, 0, prev_proc, 0, MPI_COMM_WORLD, &status);
		}

		// multiply stripes
		for(i = 0; i < a_part_size; ++i) {
			opt_A = &(buf_A[i*size]);
			opt_C = &(buf_C[i*size + ind_shift]);
			for (j = 0; j < b_part_size; ++j) {
				opt_B = &(buf_B[j*size]);
				tmp_accumulator = 0.0;
				for(k = 0; k < size; ++k) 
					tmp_accumulator += opt_A[k] * opt_B[k];
				
				opt_C[j] = tmp_accumulator;
			}
		}

		// update index shift
		ind_shift -= size/num_processes;
		if  ((rank - (p + 1) + num_processes) % num_processes < size % num_processes) 
			ind_shift--;
		if (ind_shift < 0)
			ind_shift += size;
	}
	
	// gather result	
	MPI_Gatherv(buf_C, a_part_size * size, MPI_FLOAT, C, part_num_elems, displs, MPI_FLOAT, 0, MPI_COMM_WORLD);
	
	// coordinator: free resources
	if (rank == 0) {
		transpose(B, size);
		free(part_num_elems);
		free(displs);
	}

	// free memory
	free(buf_A);
	free(buf_B);
	free(buf_C);
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

