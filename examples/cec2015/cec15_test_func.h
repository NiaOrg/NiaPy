#ifndef CEC15_TEST_FUNCTIONS_H
#define CEC15_TEST_FUNCTIONS_H

#define INF 1.0e99
#define EPS 1.0e-14
#define E  2.7182818284590452353602874713526625
#define PI 3.1415926535897932384626433832795029

#define MAX_OF_RUNS 20
#define RECORDING_POINTS_NUM 19
#define MAX_FUNCTION_NUMBER  15
#define TIMES_OF_EVAL  50
#define DIMS 2

int get_number_of_run();
void record(int number_of_run, int func_number, int dim, double* x, double fx);

void sphere_func (double *, double *, int , double *,double *, int, int); /* Sphere */
void ellips_func(double *, double *, int , double *,double *, int, int); /* Ellipsoidal */
void bent_cigar_func(double *, double *, int , double *,double *, int, int); /* Discus */
void discus_func(double *, double *, int , double *,double *, int, int);  /* Bent_Cigar */
void dif_powers_func(double *, double *, int , double *,double *, int, int);  /* Different Powers */
void rosenbrock_func (double *, double *, int , double *,double *, int, int); /* Rosenbrock's */
void schaffer_F7_func (double *, double *, int , double *,double *, int, int); /* Schwefel's F7 */
void ackley_func (double *, double *, int , double *,double *, int, int); /* Ackley's */
void rastrigin_func (double *, double *, int , double *,double *, int, int); /* Rastrigin's  */
void weierstrass_func (double *, double *, int , double *,double *, int, int); /* Weierstrass's  */
void griewank_func (double *, double *, int , double *,double *, int, int); /* Griewank's  */
void schwefel_func (double *, double *, int , double *,double *, int, int); /* Schwefel's */
void katsuura_func (double *, double *, int , double *,double *, int, int); /* Katsuura */
void bi_rastrigin_func (double *, double *, int , double *,double *, int, int); /* Lunacek Bi_rastrigin */
void grie_rosen_func (double *, double *, int , double *,double *, int, int); /* Griewank-Rosenbrock  */
void escaffer6_func (double *, double *, int , double *,double *, int, int); /* Expanded Scaffer??s F6  */
void step_rastrigin_func (double *, double *, int , double *,double *, int, int); /* Noncontinuous Rastrigin's  */
void happycat_func (double *, double *, int , double *,double *, int, int); /* HappyCat */
void hgbat_func (double *, double *, int , double *,double *, int, int); /* HGBat  */

void hf01 (double*, double*, int, double*, double*, int*, int, int); /* Hybrid Function 1 */
void hf02 (double*, double*, int, double*, double*, int*, int, int); /* Hybrid Function 2 */
void hf03 (double*, double*, int, double*, double*, int*, int, int); /* Hybrid Function 3 */
void hf04 (double*, double*, int, double*, double*, int*, int, int); /* Hybrid Function 4 */
void hf05 (double*, double*, int, double*, double*, int*, int, int); /* Hybrid Function 5 */
void hf06 (double*, double*, int, double*, double*, int*, int, int); /* Hybrid Function 6 */

void cf01 (double*, double*, int, double*, double*, int); /* Composition Function 1 */
void cf02 (double*, double*, int, double*, double*, int); /* Composition Function 2 */
void cf03 (double*, double*, int, double*, double*, int); /* Composition Function 3 */
void cf04 (double*, double*, int, double*, double*, int); /* Composition Function 4 */
void cf05 (double*, double*, int, double*, double*, int); /* Composition Function 5 */
void cf06 (double*, double*, int, double*, double*, int); /* Composition Function 6 */
void cf07 (double*, double*, int, double*, double*, int*, int); /* Composition Function 7 */
void cf08 (double*, double*, int, double*, double*, int*, int); /* Composition Function 8 */

void shiftfunc (double*,double*,int,double*);
void rotatefunc (double*,double*,int, double*);
void sr_func (double*, double*, int, double*, double*, double, int, int); /* shift and rotate */
void asyfunc (double*, double*, int, double);
void oszfunc (double*, double*, int);
void cf_cal(double*, double*, int, double*,double*,double*,double*,int);

/*
 * evaluate specific function
 */
double* cec15_test_func(double *x, double *f, int nx, int mx,int func_num);
double runtest(double*, int, int);

#ifndef NO_RECORDING
/*
 *  Number of run
 */
void set_number_of_run(int run);
/* 
 * output to file
 */
// dir can only like test/resultdir
void write_result_statistics_to_file(char* dir, char * file_prefix);

#endif //NO_RECORDING

#endif //CEC15_TEST_FUNCTIONS_H
