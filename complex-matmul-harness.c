/* Test and timing harness program for developing a dense matrix
   multiplication routine for the CS3014 module */

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <assert.h>
#include <omp.h>
#include <math.h>
#include <x86intrin.h>
#include <limits.h>




/* the following two definitions of DEBUGGING control whether or not
   debugging information is written out. To put the program into
   debugging mode, uncomment the following line: */
/*#define DEBUGGING(_x) _x */
/* to stop the printing of debugging information, use the following line: */
#define DEBUGGING(_x)

struct complex {
  float real;
  float imag;
};

/* write matrix to stdout */
void write_out(struct complex ** a, int dim1, int dim2)
{
  int i, j;

  for ( i = 0; i < dim1; i++ ) {
    for ( j = 0; j < dim2 - 1; j++ ) {
      printf("%.3f + %.3fi ", a[i][j].real, a[i][j].imag);
    }
    printf("%.3f + %.3fi\n", a[i][dim2-1].real, a[i][dim2-1].imag);
  }
}


/* create new empty matrix */
struct complex ** new_empty_matrix(int dim1, int dim2)
{
  struct complex ** result = malloc(sizeof(struct complex*) * dim1);
  struct complex * new_matrix = malloc(sizeof(struct complex) * dim1 * dim2);
  int i;

  for ( i = 0; i < dim1; i++ ) {
    result[i] = &(new_matrix[i*dim2]);
  }

  return result;
}

void free_matrix(struct complex ** matrix) {
  free (matrix[0]); /* free the contents */
  free (matrix); /* free the header */
}

/* take a copy of the matrix and return in a newly allocated matrix */
struct complex ** copy_matrix(struct complex ** source_matrix, int dim1, int dim2)
{
  int i, j;
  struct complex ** result = new_empty_matrix(dim1, dim2);

  for ( i = 0; i < dim1; i++ ) {
    for ( j = 0; j < dim2; j++ ) {
      result[i][j] = source_matrix[i][j];
    }
  }

  return result;
}

/* create a matrix and fill it with random numbers */
struct complex ** gen_random_matrix(int dim1, int dim2)
{
  const int random_range = 512; // constant power of 2
  struct complex ** result;
  int i, j;
  struct timeval seedtime;
  int seed;

  result = new_empty_matrix(dim1, dim2);

  /* use the microsecond part of the current time as a pseudorandom seed */
  gettimeofday(&seedtime, NULL);
  seed = seedtime.tv_usec;
  srandom(seed);

  /* fill the matrix with random numbers */
  for ( i = 0; i < dim1; i++ ) {
    for ( j = 0; j < dim2; j++ ) {
      /* evenly generate values in the range [0, random_range-1)*/
      result[i][j].real = (float)(random() % random_range);
      result[i][j].imag = (float)(random() % random_range);

      /* at no loss of precision, negate the values sometimes */
      /* so the range is now (-(random_range-1), random_range-1)*/
      if (random() & 1) result[i][j].real = -result[i][j].real;
      if (random() & 1) result[i][j].imag = -result[i][j].imag;
    }
  }

  return result;
}

/* check the sum of absolute differences is within reasonable epsilon */
/* returns number of differing values */
void check_result(struct complex ** result, struct complex ** control, int dim1, int dim2)
{
  int i, j;
  double sum_abs_diff = 0.0;
  const double EPSILON = 0.0625;

  for ( i = 0; i < dim1; i++ ) {
    for ( j = 0; j < dim2; j++ ) {
      double diff;
      diff = fabs(control[i][j].real - result[i][j].real);
      sum_abs_diff = sum_abs_diff + diff;

      diff = fabs(control[i][j].imag - result[i][j].imag);
      sum_abs_diff = sum_abs_diff + diff;
    }
  }

  if ( sum_abs_diff > EPSILON ) {
    fprintf(stderr, "WARNING: sum of absolute differences (%f) > EPSILON (%f)\n",
      sum_abs_diff, EPSILON);
  }
}

/* multiply matrix A times matrix B and put result in matrix C */
void matmul(struct complex ** A, struct complex ** B, struct complex ** C, int a_dim1, int a_dim2, int b_dim2)
{
  int i, j, k;

  for ( i = 0; i < a_dim1; i++ ) {
    for( j = 0; j < b_dim2; j++ ) {
      struct complex sum;
      sum.real = 0.0;
      sum.imag = 0.0;
      for ( k = 0; k < a_dim2; k++ ) {
        // the following code does: sum += A[i][k] * B[k][j];
        struct complex product;
        product.real = A[i][k].real * B[k][j].real - A[i][k].imag * B[k][j].imag;
        product.imag = A[i][k].real * B[k][j].imag + A[i][k].imag * B[k][j].real;
        sum.real += product.real;
        sum.imag += product.imag;
      }
      C[i][j] = sum;
    }
  }

}

/* the fast version of matmul written by the team */
void team_matmul(struct complex ** A, struct complex ** B, struct complex ** C, int a_rows, int a_cols, int b_cols) {
  //replace this
  //matmul(A, B, C, a_rows, a_cols, b_cols);
  int iterations = a_rows * a_cols;
  if (iterations < a_rows){
    iterations = INT_MAX;
  } else {
    iterations = iterations * b_cols;
    if (iterations < b_cols){
      iterations = INT_MAX;
    }
  }
  if (iterations < 27000 || (a_cols < 10 && a_rows * b_cols < 12960000) ){
    matmul(A, B, C, a_rows, a_cols, b_cols);
    return;
  }
  int b_rows = b_cols;  //transposing b
  b_cols = a_cols;

  float ** real_a = (float**) malloc(a_rows * sizeof(float*));
  float ** imag_a = (float**) malloc(a_rows * sizeof(float*));
  float ** real_b = (float**) malloc(b_rows * sizeof(float*));
  float ** imag_b = (float**) malloc(b_rows * sizeof(float*));

  // //i = a_rows
  // //k = a_cols
  // //k = b_rows
  // //j = b_cols 

  int i, j;
  for ( i = 0; i < a_rows; i++ ) {
    real_a[i] = (float*) malloc(a_cols * sizeof(float));
    imag_a[i] = (float*) malloc(a_cols * sizeof(float));   
    for( j = 0; j < a_cols; j++ ) {
      real_a[i][j] = A[i][j].real;
      imag_a[i][j] = A[i][j].imag; 
    }
  }

  for ( i = 0; i < b_rows; i++ ) {
    real_b[i] = (float*) malloc(b_cols * sizeof(float));
    imag_b[i] = (float*) malloc(b_cols * sizeof(float));  
    for( j = 0; j < b_cols; j++ ) {
      real_b[i][j] = B[j][i].real;
      imag_b[i][j] = B[j][i].imag; 
    }
  }


  //__m128 real_a_mat, real_b_mat, imag_a_mat, imag_b_mat, product_real, product_imag, sum_real, sum_imag, a, b;
  
  if (iterations < 512000000 && (iterations > 42875 || a_cols < 4)){
   #pragma omp parallel for if (a_rows > 50 && a_cols * b_cols > 10000)
    for ( i = 0; i < a_rows; i++ ) {

   #pragma omp parallel for if (iterations > 4913000 && a_cols > 10)
      for( j = 0; j < b_rows; j++ ) {
        int k;
        float sum_real, sum_imag, product_real, product_imag;
        sum_real = 0.0;
        sum_imag = 0.0;
        for ( k = 0; k < a_cols; k++ ) { //a_cols = b_cols
          // the following code does: sum += A[i][k] * B[k][j];
          
          product_real = real_a[i][k] * real_b[j][k] - imag_a[i][k] * imag_b[j][k];
          product_imag = real_a[i][k] * imag_b[j][k] + imag_a[i][k] * real_b[j][k];
          sum_real += product_real;
          sum_imag += product_imag;
        }
        C[i][j].real = sum_real;
        C[i][j].imag = sum_imag;
      }
    }
  } else {
    int a_cols_limit, last_a_col, extra;
    float real_limit_total, imag_limit_total, real_extra, imag_extra;
    real_limit_total = 0;
    imag_limit_total = 0; 
    real_extra = 0; 
    imag_extra = 0;
    extra = a_cols % 4;
    a_cols_limit = a_cols - extra;
    #pragma omp parallel for if(a_rows > 50 && a_cols * b_cols > 10000) //WHEEEEEEEEEEEEEEEEEE
    for ( i = 0; i < a_rows; i++ ) {
     #pragma omp parallel for if(iterations > 4913000 && a_cols > 10)
      for( j = 0; j < b_rows; j++ ) {
        // float addhoc_real[4];
        // float addhoc_imag[4];
        int k;
        __m128 sum_real, sum_imag;
        sum_real = _mm_set1_ps(0.0);
        sum_imag = sum_real;
        __m128 real_a_mat, real_b_mat, imag_a_mat, imag_b_mat, product_real, product_imag, a, b;

        for ( k = 0; k < a_cols_limit; k+=4 ) { //a_cols = b_cols
          // the following code does: sum += A[i][k] * B[k][j];
          real_a_mat = _mm_load_ps(&real_a[i][k]);
          real_b_mat = _mm_load_ps(&real_b[j][k]);
          imag_a_mat = _mm_load_ps(&imag_a[i][k]);
          imag_b_mat = _mm_load_ps(&imag_b[j][k]);
          a = _mm_mul_ps(real_a_mat, real_b_mat);
          b = _mm_mul_ps(imag_a_mat, imag_b_mat);
          product_real = _mm_sub_ps(a,b);
          a = _mm_mul_ps(real_a_mat, imag_b_mat);
          b = _mm_mul_ps(imag_a_mat, real_b_mat);
          product_imag = _mm_add_ps(a,b);
          sum_real = _mm_add_ps(sum_real, product_real);
          sum_imag = _mm_add_ps(sum_imag, product_imag);
        }
        sum_real =  _mm_hadd_ps(sum_real, sum_real);
        sum_real = _mm_hadd_ps(sum_real, sum_real);
        C[i][j].real += ((float*)&sum_real)[0];
        // _mm_store_ps(addhoc_real, sum_real);
        // C[i][j].real += addhoc_real[0] + addhoc_real[1] + addhoc_real[2] + addhoc_real[3];
        sum_imag = _mm_hadd_ps(sum_imag, sum_imag);
        sum_imag = _mm_hadd_ps(sum_imag, sum_imag);
        C[i][j].imag += ((float*)&sum_imag)[0];
        // _mm_store_ps(addhoc_imag, sum_imag);
        // C[i][j].imag += addhoc_imag[0] + addhoc_imag[1] + addhoc_imag[2] + addhoc_imag[3];

        for (k = a_cols_limit; k <a_cols; k++){
          C[i][j].real += real_a[i][k] * real_b[j][k] - imag_a[i][k] * imag_b[j][k];
          C[i][j].imag +=  real_a[i][k] * imag_b[j][k] + imag_a[i][k] * real_b[j][k];
        }
      }
    }
  }
  

  


  


  
}

long long time_diff(struct timeval * start, struct timeval * end) {
  return (end->tv_sec - start->tv_sec) * 1000000L + (end->tv_usec - start->tv_usec);
}

int main(int argc, char ** argv)
{
  struct complex ** A, ** B, ** C;
  struct complex ** control_matrix;
  long long control_time, mul_time;
  double speedup;
  int a_dim1, a_dim2, b_dim1, b_dim2, errs;
  struct timeval pre_time, start_time, stop_time;

  if ( argc != 5 ) {
    fprintf(stderr, "Usage: matmul-harness <A nrows> <A ncols> <B nrows> <B ncols>\n");
    exit(1);
  }
  else {
    a_dim1 = atoi(argv[1]);
    a_dim2 = atoi(argv[2]);
    b_dim1 = atoi(argv[3]);
    b_dim2 = atoi(argv[4]);
  }

  /* check the matrix sizes are compatible */
  if ( a_dim2 != b_dim1 ) {
    fprintf(stderr,
      "FATAL number of columns of A (%d) does not match number of rows of B (%d)\n",
      a_dim2, b_dim1);
    exit(1);
  }

  /* allocate the matrices */
  A = gen_random_matrix(a_dim1, a_dim2);
  B = gen_random_matrix(b_dim1, b_dim2);
  C = new_empty_matrix(a_dim1, b_dim2);
  control_matrix = new_empty_matrix(a_dim1, b_dim2);

  DEBUGGING( {
    printf("matrix A:\n");
    write_out(A, a_dim1, a_dim2);
    printf("\nmatrix B:\n");
    write_out(A, a_dim1, a_dim2);
    printf("\n");
  } )

  /* record control start time */
  gettimeofday(&pre_time, NULL);


  /* use a simple matmul routine to produce control result */
  matmul(A, B, control_matrix, a_dim1, a_dim2, b_dim2);

  /* record starting time */
  gettimeofday(&start_time, NULL);

  /* perform matrix multiplication */
  team_matmul(A, B, C, a_dim1, a_dim2, b_dim2);

  /* record finishing time */
  gettimeofday(&stop_time, NULL);

  /* compute elapsed times and speedup factor */
  control_time = time_diff(&pre_time, &start_time);
  mul_time = time_diff(&start_time, &stop_time);
  speedup = (float) control_time / mul_time;

  printf("Matmul time: %lld microseconds\n", mul_time);
  printf("control time : %lld microseconds\n", control_time);
  if (mul_time > 0 && control_time > 0) {
    printf("speedup: %.2fx\n", speedup);
  }

  /* now check that the team's matmul routine gives the same answer
     as the known working version */
  check_result(C, control_matrix, a_dim1, b_dim2);

  /* free all matrices */
  free_matrix(A);
  free_matrix(B);
  free_matrix(C);
  free_matrix(control_matrix);

  return 0;
}
