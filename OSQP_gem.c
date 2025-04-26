#include <stdio.h>
#include <stdlib.h>
#include "osqp.h"
#include <math.h> // For OSQP_INFTY

int main(int argc, char **argv) {
    // Problem dimensions
    OSQPInt n = 15; // n_features + n_samples
    OSQPInt m = 20; // 2 * n_samples

    // P matrix data (CSC, upper triangular)
    OSQPFloat P_x[25] = { 5.90891706, -1.02306007,  8.84763601,  1.5480568,  -0.69732452,  6.28283379,
                          1.89707801, -0.57469652,  2.57115414, 12.50741694, -0.03808408,  5.70506189,
                         -2.83245226,  3.63001971, 11.71025262,  1.,          1.,          1.,
                          1.,          1.,          1.,          1.,          1.,          1.,
                          1.        };
    OSQPInt P_nnz = 25;
    OSQPInt P_i[25] = { 0,  0,  1,  0,  1,  2,  0,  1,  2,  3,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13,
                       14};
    OSQPInt P_p[16] = { 0,  1,  3,  6, 10, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25};

    // q vector
    OSQPFloat q[15] = { -4.29478866,  30.85070497,  -1.50726179, -12.41532811,  26.39931215,
                         1.,           1.,           1.,           1.,           1.,
                         1.,           1.,           1.,           1.,           1.        };

    // A matrix data (CSC)
    OSQPFloat A_x[120] = { 0.49671415, -0.23413696, -0.46341769, -0.56228753,  1.46564877,  0.11092259,
                         -0.60170661, -1.22084365,  0.73846658, -0.71984421, -0.49671415,  0.23413696,
                          0.46341769,  0.56228753, -1.46564877, -0.11092259,  0.60170661,  1.22084365,
                         -0.73846658,  0.71984421, -0.1382643,   1.57921282, -0.46572975, -1.01283112,
                         -0.2257763,  -1.15099358,  1.85227818,  0.2088636,   0.17136828, -0.46063877,
                          0.1382643,  -1.57921282,  0.46572975,  1.01283112,  0.2257763,   1.15099358,
                         -1.85227818, -0.2088636,  -0.17136828,  0.46063877,  0.64768854,  0.76743473,
                          0.24196227,  0.31424733,  0.0675282,   0.37569802, -0.01349722, -1.95967012,
                         -0.11564828,  1.05712223, -0.64768854, -0.76743473, -0.24196227, -0.31424733,
                         -0.0675282,  -0.37569802,  0.01349722,  1.95967012,  0.11564828, -1.05712223,
                          1.52302986, -0.46947439, -1.91328024, -0.90802408, -1.42474819, -0.60063869,
                         -1.05771093, -1.32818605, -0.3011037,   0.34361829, -1.52302986,  0.46947439,
                          1.91328024,  0.90802408,  1.42474819,  0.60063869,  1.05771093,  1.32818605,
                          0.3011037,  -0.34361829, -0.23415337,  0.54256004, -1.72491783, -1.4123037,
                         -0.54438272, -0.29169375,  0.82254491,  0.19686124, -1.47852199, -1.76304016,
                          0.23415337, -0.54256004,  1.72491783,  1.4123037,   0.54438272,  0.29169375,
                         -0.82254491, -0.19686124,  1.47852199,  1.76304016, -1.,         -1.,
                         -1.,         -1.,         -1.,         -1.,         -1.,         -1.,
                         -1.,         -1.,         -1.,         -1.,         -1.,         -1.,
                         -1.,         -1.,         -1.,         -1.,         -1.,         -1.        };
    OSQPInt A_nnz = 120;
    OSQPInt A_i[120] = { 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,  0,  1,  2,  3,
                         4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,  0,  1,  2,  3,  4,  5,  6,  7,
                         8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11,
                        12, 13, 14, 15, 16, 17, 18, 19,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,
                        16, 17, 18, 19,  0, 10,  1, 11,  2, 12,  3, 13,  4, 14,  5, 15,  6, 16,  7, 17,  8, 18,  9, 19};
    OSQPInt A_p[16] = {  0,  20,  40,  60,  80, 100, 102, 104, 106, 108, 110, 112, 114, 116, 118, 120};

    // Lower and upper bounds
    OSQPFloat l[20] = {-OSQP_INFTY, -OSQP_INFTY, -OSQP_INFTY, -OSQP_INFTY, -OSQP_INFTY, -OSQP_INFTY, -OSQP_INFTY, -OSQP_INFTY, -OSQP_INFTY, -OSQP_INFTY,
                       -OSQP_INFTY, -OSQP_INFTY, -OSQP_INFTY, -OSQP_INFTY, -OSQP_INFTY, -OSQP_INFTY, -OSQP_INFTY, -OSQP_INFTY, -OSQP_INFTY, -OSQP_INFTY};
    OSQPFloat u[20] = { 5.98688476, -5.65315541,  3.88294907,  0.17630057, -0.56433632,  1.42378878,
                       -9.02240275,  1.41498176,  1.50580593,  3.12317882, -5.98688476,  5.65315541,
                       -3.88294907, -0.17630057,  0.56433632, -1.42378878,  9.02240275, -1.41498176,
                       -1.50580593, -3.12317882};

    // OSQP Variables
    OSQPInt exitflag = 0;
    OSQPSolver *solver = NULL;
    OSQPSettings *settings = NULL;
    OSQPCscMatrix* P = NULL;
    OSQPCscMatrix* A = NULL;
    FILE* f = NULL;

    // Create CSC matrices
    P = OSQPCscMatrix_new(n, n, P_nnz, P_x, P_i, P_p);
    A = OSQPCscMatrix_new(m, n, A_nnz, A_x, A_i, A_p);

    // Create settings
    settings = OSQPSettings_new();

    // Check if allocations were successful
    if (!P || !A || !settings) {
        fprintf(stderr, "Error: Failed to allocate OSQP structures\n");
        exitflag = 1;
        goto cleanup;
    }

    // Set verbose setting
    settings->verbose = 1;

    // Setup solver
    exitflag = osqp_setup(&solver, P, q, A, l, u, m, n, settings);
    if (exitflag) {
        fprintf(stderr, "Error: OSQP setup failed with error code %d\n", (int)exitflag);
        exitflag = 1; // Ensure non-zero exit code on setup failure
        goto cleanup;
    }

    // Solve problem
    exitflag = osqp_solve(solver);
    if (exitflag) {
        fprintf(stderr, "Error: OSQP solve failed with error code %d\n", (int)exitflag);
        exitflag = 1; // Ensure non-zero exit code on solve failure
        goto cleanup;
    }

    // Check solver status and print results
    if (solver && solver->info->status_val == OSQP_SOLVED) {
        printf("Huber estimates: [%.2f, %.2f, %.2f, %.2f, %.2f]\n",
               solver->solution->x[0], solver->solution->x[1], solver->solution->x[2],
               solver->solution->x[3], solver->solution->x[4]);

        printf("Optimal objective value: %.16f\n", solver->info->obj_val);

        // Save optimal value to file
        f = fopen("c_results.txt", "w");
        if (f) {
            fprintf(f, "%.16f\n", solver->info->obj_val);
            fclose(f);
            f = NULL; // Avoid double closing in cleanup
        } else {
            fprintf(stderr, "Error: Could not open file c_results.txt for writing results\n");
            exitflag = 1; // Indicate error
        }
    } else if (solver) {
        fprintf(stderr, "Error: Solver failed with status: %s\n", solver->info->status);
        exitflag = 1; // Indicate error
    } else {
        fprintf(stderr, "Error: Solver initialization failed.\n");
        exitflag = 1; // Indicate error
    }

cleanup:
    // Close file if it's still open (e.g., due to an error after opening)
    if (f) fclose(f);

    // Cleanup OSQP structures
    if (solver) osqp_cleanup(solver);
    if (A) OSQPCscMatrix_free(A);
    if (P) OSQPCscMatrix_free(P);
    if (settings) OSQPSettings_free(settings);

    return (int)exitflag;
}