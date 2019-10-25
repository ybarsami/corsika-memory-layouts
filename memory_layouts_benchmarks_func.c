/**
 * Memory layout testing for CORSIKA in 3d.
 *
 * Contact:
 *   Yann Barsamian <yann.barsamian@ulb.ac.be>
 *
 * Compile:
 *   gcc memory_layouts_benchmarks_func.c -lm -O3 -ffast-math -fopenmp -march=native -std=gnu11 -Wall -o test.out -fopt-info-vec-all 2> vect_info_func.txt
 *   icc memory_layouts_benchmarks_func.c -lm -O3             -qopenmp -march=native -std=gnu11 -Wall -o test.out -qopt-report=5 -qopt-report-phase=vec
 *   (-ffast-math is needed with gcc to vectorize the logf function call)
 */

#include <math.h>            // function  logf --- the C99 macro 'log' should detect that we are using floats and convert automatically a log(XXX) to a logf(XXX), but just to be sure.
#ifndef _XOPEN_SOURCE
#  define _XOPEN_SOURCE      // To enable drand48 and friends.
#endif
#include <omp.h>             // function  omp_get_wtime
#include <stdlib.h>          // functions posix_memalign, free ((de)allocate memory)
                             //           drand48, srand48 (random number generator)
                             //           exit (error handling)
                             // constant  EXIT_FAILURE (error handling)
                             // type      size_t
#include <stdio.h>           // function  printf

/*****************************************************************************
 *                             Simulation parameters                         *
 *****************************************************************************/

#ifndef NB_PARTICLES
#   define NB_PARTICLES 20000000 // Number of particles.
#endif
#ifndef DELTA_T
#   define DELTA_T 0.1f          // Time step (f ensures this is treated as a float for operations).
#endif
#ifndef SPAN_SIZE
#   define SPAN_SIZE 256         // Number of particles in a span of particles.
#endif
#ifndef RANDOM_SEED
#   define RANDOM_SEED 123456    // The seed used (to make sure all particles are equal).
#endif



/*****************************************************************************
 *                                Helper function                            *
 *****************************************************************************/

/*
 * Computes the square of a float.
 */
float sqr(float a) {
    return (a * a);
}

/*
 * Computes the minimum of two ints.
 */
int min(int a, int b) {
    return a < b ? a : b;
}



/*****************************************************************************
 *                                 Vectorization                             *
 *****************************************************************************/

// If the architecture supports vector size 512 (AVX-512 instructions)
#if defined(__AVX512F__)
#    define VEC_ALIGN 64
// else if it supports vector size 256 (AVX instructions)
#elif defined(__AVX__)
#    define VEC_ALIGN 32
// else probably it supports vector size 128 (SSE instructions).
#else
#    define VEC_ALIGN 16
#endif



/*****************************************************************************
 *                              Memory layout 1                              *
 *                            Array of structures                            *
 * pid_0, x_0, y_0, z_0, ... pid_1, x_1, y_1, z_1, ... pid_n, x_n, y_n, z_n  *
 *****************************************************************************/

typedef struct particle_3d {
    int pid;          // charge
    float x, y, z;    // position
    float px, py, pz; // momentum
    float t;          // time
    float e;          // energy
} particle_3d;

/*
 * Initializes array of num_particle random particles following a uniform distribution
 * for positions and momentums (uniform inside [-1; 1[^3).
 */
void create_particles_3d_aos(unsigned int num_particle, particle_3d** particles) {
    // Random seed initialization
    srand48(RANDOM_SEED);
    
    const float x_min = -1.;
    const float y_min = -1.;
    const float z_min = -1.;
    const float px_min = -1.;
    const float py_min = -1.;
    const float pz_min = -1.;
    const float x_max = 1.;
    const float y_max = 1.;
    const float z_max = 1.;
    const float px_max = 1.;
    const float py_max = 1.;
    const float pz_max = 1.;
    const float x_range = x_max - x_min;
    const float y_range = y_max - y_min;
    const float z_range = z_max - z_min;
    const float px_range = px_max - px_min;
    const float py_range = py_max - py_min;
    const float pz_range = pz_max - pz_min;
    float x, y, z, px, py, pz;
    
    for (size_t i = 0; i < num_particle; i++) {
        x = x_range * drand48() + x_min;
        y = y_range * drand48() + y_min;
        z = z_range * drand48() + z_min;
        px = px_range * drand48() + px_min;
        py = py_range * drand48() + py_min;
        pz = pz_range * drand48() + pz_min;
        (*particles)[i] = (particle_3d) {
            .pid = i % 3 - 1, // make 1/3 of particles neutral
            .x   = x,
            .y   = y,
            .z   = z,
            .px  = px,
            .py  = py,
            .pz  = pz,
            .t   = 0.,
            .e   = 0.};
    }
}

/*
 * Computes the squared momentum of a particle.
 */
float momentum_squared_aos(particle_3d part) {
    return sqr(part.px) + sqr(part.py) + sqr(part.pz);
}

float charge_aos(particle_3d part) {
    // this could be a complicated function or a lookup table
    return (float)(part.pid != 0);
}

void energy_loss_aos(particle_3d* part) {
    float beta_2 = momentum_squared_aos(*part) / sqr(part->e);
    // compute energy loss, ignoring all constants
    float energy_loss = charge_aos(*part) * (logf(beta_2 / (1.f - beta_2)) / beta_2 - 1.f);
    part->e -= energy_loss;
}

void move_particle_aos(particle_3d* p) {
    p->x += p->px * DELTA_T;
    p->y += p->py * DELTA_T;
    p->z += p->pz * DELTA_T;
    p->t += DELTA_T;
}



/*****************************************************************************
 *                              Memory layout 2                              *
 *                            Structure of arrays                            *
 * pid_0, pid_1, ... pid_n, x_0, x_1, ... x_n, y_1, y_2, ... y_n, ...        *
 *****************************************************************************/

/*
 * Initializes arrays of num_particle random particles following a uniform distribution
 * for positions and momentums (uniform inside [-1; 1[^3).
 */
void create_particles_3d_soa(unsigned int num_particle,
        int** pids, float** xs, float** ys, float** zs,
        float** pxs, float** pys, float** pzs, float** ts, float** es) {
    // Random seed initialization
    srand48(RANDOM_SEED);
    
    const float x_min = -1.;
    const float y_min = -1.;
    const float z_min = -1.;
    const float px_min = -1.;
    const float py_min = -1.;
    const float pz_min = -1.;
    const float x_max = 1.;
    const float y_max = 1.;
    const float z_max = 1.;
    const float px_max = 1.;
    const float py_max = 1.;
    const float pz_max = 1.;
    const float x_range = x_max - x_min;
    const float y_range = y_max - y_min;
    const float z_range = z_max - z_min;
    const float px_range = px_max - px_min;
    const float py_range = py_max - py_min;
    const float pz_range = pz_max - pz_min;
    float x, y, z, px, py, pz;
    
    for (size_t i = 0; i < num_particle; i++) {
        x = x_range * drand48() + x_min;
        y = y_range * drand48() + y_min;
        z = z_range * drand48() + z_min;
        px = px_range * drand48() + px_min;
        py = py_range * drand48() + py_min;
        pz = pz_range * drand48() + pz_min;
        (*pids)[i] = i % 3 - 1, // make 1/3 of particles neutral
        (*xs)[i]   = x;
        (*ys)[i]   = y;
        (*zs)[i]   = z;
        (*pxs)[i]  = px;
        (*pys)[i]  = py;
        (*pzs)[i]  = pz;
        (*ts)[i]   = 0.;
        (*es)[i]   = 0.;
    }
}

/*
 * Computes the squared momentum of a particle.
 */
float momentum_squared_soa(float px, float py, float pz) {
    return sqr(px) + sqr(py) + sqr(pz);
}

float charge_soa(int pid) {
    // this could be a complicated function or a lookup table
    return (float)(pid != 0);
}

// Only one pointer here, no need to use the "restrict" keyword.
void energy_loss_soa(float pid, float px, float py, float pz, float* e) {
    float beta_2 = momentum_squared_soa(px, py, pz) / sqr(*e);
    // compute energy loss, ignoring all constants
    float energy_loss = charge_soa(pid) * (logf(beta_2 / (1.f - beta_2)) / beta_2 - 1.f);
    *e -= energy_loss;
}

// The "restrict" keyword asserts that the data pointed to by x, y, z, and t are not accessed via any other pointer --- otherwise, the compiler does not auto-vectorize the loop; adding a '#pragma GCC ivdep' would also work.
void move_particle_soa(float px, float py, float pz, float* restrict x, float* restrict y, float* restrict z, float* restrict t) {
    *x += px * DELTA_T;
    *y += py * DELTA_T;
    *z += pz * DELTA_T;
    *t += DELTA_T;
}



/*****************************************************************************
 *                                  Benchmarks                               *
 * REMARK: The two operations chosen here (energy_loss and move_particle)    *
 * are both vectorizable. It would be useful to have a more realistic        *
 * benchmark with some operations that can be vectorized, and some that      *
 * cannot.                                                                   *
 *****************************************************************************/

int main(int argc, char** argv) {
    // Timing
    double time_start, time_simu;
    
    // Memory allocation for particles (array of structures). REMARK: Alignment is not useful when using loops with that much iterations.
    particle_3d* particles;
    posix_memalign((void**)&particles, VEC_ALIGN, NB_PARTICLES * sizeof(particle_3d));
    
    // Memory allocation for particles (structure of arrays). REMARK: Alignment is not useful when using loops with that much iterations.
    int* pids;
    posix_memalign((void**)&pids, VEC_ALIGN, NB_PARTICLES * sizeof(int));
    float* xs;
    posix_memalign((void**)&xs, VEC_ALIGN, NB_PARTICLES * sizeof(float));
    float* ys;
    posix_memalign((void**)&ys, VEC_ALIGN, NB_PARTICLES * sizeof(float));
    float* zs;
    posix_memalign((void**)&zs, VEC_ALIGN, NB_PARTICLES * sizeof(float));
    float* pxs;
    posix_memalign((void**)&pxs, VEC_ALIGN, NB_PARTICLES * sizeof(float));
    float* pys;
    posix_memalign((void**)&pys, VEC_ALIGN, NB_PARTICLES * sizeof(float));
    float* pzs;
    posix_memalign((void**)&pzs, VEC_ALIGN, NB_PARTICLES * sizeof(float));
    float* ts;
    posix_memalign((void**)&ts, VEC_ALIGN, NB_PARTICLES * sizeof(float));
    float* es;
    posix_memalign((void**)&es, VEC_ALIGN, NB_PARTICLES * sizeof(float));
    
    // Method AOS_1: one loop over particles (vectorized, with stride 9)
    create_particles_3d_aos(NB_PARTICLES, &particles);
    time_start = omp_get_wtime();
    for (size_t i = 0; i < NB_PARTICLES; i++) {
        energy_loss_aos(&particles[i]);
        move_particle_aos(&particles[i]);
    }
    time_simu = (double) (omp_get_wtime() - time_start);
    printf("Execution time (AoS, one loop)                              : %g s\n", time_simu);
    
    // Method AOS_2: two nested loops over particles (inner loop vectorized, with stride 9)
    // REMARK: same order on operation as AOS_1, the code is less efficient because there is one prologue and one epilogue each time we reach the inner loop
    create_particles_3d_aos(NB_PARTICLES, &particles);
    time_start = omp_get_wtime();
    for (size_t big_i = 0; big_i < NB_PARTICLES; big_i += SPAN_SIZE) {
        for (size_t i = big_i; i < min(NB_PARTICLES, big_i + SPAN_SIZE); i++) {
            energy_loss_aos(&particles[i]);
            move_particle_aos(&particles[i]);
        }
    }
    time_simu = (double) (omp_get_wtime() - time_start);
    printf("Execution time (AoS, two nested loops)                      : %g s\n", time_simu);
    
    // Method AOS_1A: loop fission, one loop per operation over all particles (vectorized, with stride 9)
    create_particles_3d_aos(NB_PARTICLES, &particles);
    time_start = omp_get_wtime();
    for (size_t i = 0; i < NB_PARTICLES; i++) {
        energy_loss_aos(&particles[i]);
    }
    for (size_t i = 0; i < NB_PARTICLES; i++) {
        move_particle_aos(&particles[i]);
    }
    time_simu = (double) (omp_get_wtime() - time_start);
    printf("Execution time (AoS, one loop + loop fission)               : %g s\n", time_simu);
    
    // Method AOS_2A: one loop nest with inner loop fission, one inner loop per operation (vectorized, with stride 9)
    create_particles_3d_aos(NB_PARTICLES, &particles);
    time_start = omp_get_wtime();
    for (size_t big_i = 0; big_i < NB_PARTICLES; big_i += SPAN_SIZE) {
        for (size_t i = big_i; i < min(NB_PARTICLES, big_i + SPAN_SIZE); i++) {
            energy_loss_aos(&particles[i]);
        }
        for (size_t i = big_i; i < min(NB_PARTICLES, big_i + SPAN_SIZE); i++) {
            move_particle_aos(&particles[i]);
        }
    }
    time_simu = (double) (omp_get_wtime() - time_start);
    printf("Execution time (AoS, two nested loops + inner loop fission) : %g s\n", time_simu);
    
    // Method SOA_1: one loop over particles (vectorized, with stride 1)
    create_particles_3d_soa(NB_PARTICLES, &pids, &xs, &ys, &zs, &pxs, &pys, &pzs, &ts, &es);
    time_start = omp_get_wtime();
    // Pragma needed: there are two function calls, so the compiler cannot know if the pointer to the array es (function call 1) overlaps with a pointer to the array xs, ys, zs or ts (function call 2); here we tell it to not worry about pointer aliasing
    #pragma GCC ivdep
    for (size_t i = 0; i < NB_PARTICLES; i++) {
        energy_loss_soa(pids[i], pxs[i], pys[i], pzs[i], &es[i]);
        move_particle_soa(pxs[i], pys[i], pzs[i], &xs[i], &ys[i], &zs[i], &ts[i]);
    }
    time_simu = (double) (omp_get_wtime() - time_start);
    printf("Execution time (SoA, one loop)                              : %g s\n", time_simu);
    
    // Method SOA_2: two nested loops over particles (inner loop vectorized, with stride 1)
    // REMARK: same order on operation as SOA_1, the code is less efficient because there is one prologue and one epilogue each time we reach the inner loop
    create_particles_3d_soa(NB_PARTICLES, &pids, &xs, &ys, &zs, &pxs, &pys, &pzs, &ts, &es);
    time_start = omp_get_wtime();
    for (size_t big_i = 0; big_i < NB_PARTICLES; big_i += SPAN_SIZE) {
        // Pragma needed: there are two function calls, so the compiler cannot know if the pointer to the array es (function call 1) overlaps with a pointer to the array xs, ys, zs or ts (function call 2); here we tell it to not worry about pointer aliasing
        #pragma GCC ivdep
        for (size_t i = big_i; i < min(NB_PARTICLES, big_i + SPAN_SIZE); i++) {
            energy_loss_soa(pids[i], pxs[i], pys[i], pzs[i], &es[i]);
            move_particle_soa(pxs[i], pys[i], pzs[i], &xs[i], &ys[i], &zs[i], &ts[i]);
        }
    }
    time_simu = (double) (omp_get_wtime() - time_start);
    printf("Execution time (SoA, two nested loops)                      : %g s\n", time_simu);
    
    // Method SOA_1A: loop fission, one loop per operation over all particles (vectorized, with stride 1)
    create_particles_3d_soa(NB_PARTICLES, &pids, &xs, &ys, &zs, &pxs, &pys, &pzs, &ts, &es);
    time_start = omp_get_wtime();
    for (size_t i = 0; i < NB_PARTICLES; i++) {
        energy_loss_soa(pids[i], pxs[i], pys[i], pzs[i], &es[i]);
    }
    for (size_t i = 0; i < NB_PARTICLES; i++) {
        move_particle_soa(pxs[i], pys[i], pzs[i], &xs[i], &ys[i], &zs[i], &ts[i]);
    }
    time_simu = (double) (omp_get_wtime() - time_start);
    printf("Execution time (SoA, one loop + loop fission)               : %g s\n", time_simu);
    
    // Method SOA_2A: one loop nest with inner loop fission, one inner loop per operation (vectorized, with stride 1)
    create_particles_3d_soa(NB_PARTICLES, &pids, &xs, &ys, &zs, &pxs, &pys, &pzs, &ts, &es);
    time_start = omp_get_wtime();
    for (size_t big_i = 0; big_i < NB_PARTICLES; big_i += SPAN_SIZE) {
        for (size_t i = big_i; i < min(NB_PARTICLES, big_i + SPAN_SIZE); i++) {
            energy_loss_soa(pids[i], pxs[i], pys[i], pzs[i], &es[i]);
        }
        for (size_t i = big_i; i < min(NB_PARTICLES, big_i + SPAN_SIZE); i++) {
            move_particle_soa(pxs[i], pys[i], pzs[i], &xs[i], &ys[i], &zs[i], &ts[i]);
        }
    }
    time_simu = (double) (omp_get_wtime() - time_start);
    printf("Execution time (SoA, two nested loops + inner loop fission) : %g s\n", time_simu);
    
    // Be clean
    free(particles);
    free(pids);
    free(xs);
    free(ys);
    free(zs);
    free(pxs);
    free(pys);
    free(pzs);
    free(ts);
    free(es);
}

