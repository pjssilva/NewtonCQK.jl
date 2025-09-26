#include <stdlib.h>
#include <stdio.h>
#include "cont_quad_knapsack.h"

/*
 * Function Name: newton_projection
 *
 * Description: Wrapper for Newton method for projection cont. quad. knapsack 
 *     problem. It should be used by code that wants to pass the vectors that 
 *     describe the problem instead of the cqk_problem structure.
 * 
 *     The problem has the form:
 *
 *     min  1/2 || x - x0 ||_2^2
 *     s.t. b'x = r
 *     low <= x <= up,
 *
 * Input:
 *     int n: dimension of the problem;
 *     double *x0: point to project;
 *     double *b: normal do the hyperplane;
 *     double r: rhs of the hyperplane;
 *     double *low: box lower bound;
 *     double *up: box upper bound.
 * 
 * Output:
 *     double *x: solution obtained.
 * 
 * Return value: the number of iterations required to compute the
 *     solution, -1 in case of failure, -2 in case of infeasible
 *     problem.
 */
int newton_projection(int n, double *restrict x0, double *restrict b, 
    double r, double *restrict low, double *restrict up, double *x) {

    /* Find out the number of nonzero values in b */
    int nonzero = 0;
    for (unsigned i = 0; i < n; ++i)
        if (b[i] != 0.0)
            ++nonzero;

    /* Copy problem data. */
    cqk_problem prob;
    allocate_cqk_problem(nonzero, &prob);
    prob.r = r;

    unsigned j = 0;
    for (unsigned i = 0; i < n; ++i) {
        if (b[i] == 0.0)
            continue;

        prob.d[j] = 1.0;
        if (b[i] > 0.0) {
            prob.a[j] = x0[i];
            prob.b[j] = b[i];
            prob.low[j] = low[i];
            prob.up[j] = up[i];
        } 
        else {
            prob.a[j] = -x0[i];
            prob.b[j] = -b[i];
            prob.low[j] = -up[i];
            prob.up[j] = -low[i];
        }

        ++j;
    }

    double *sol = (double *) malloc(nonzero*sizeof(double));
    if (!sol) {
        fprintf(stderr, "Memory allocation error, line %d, file %s\n",
                __LINE__, __FILE__);
        exit(1);
    }

    /* Project */
    int res = newton(&prob, NULL, sol);

    /* Copy the solution back */
    j = 0;
    for (unsigned i = 0; i < n; ++i){
        if (b[i] == 0) {
            x[i] = MIN(MAX(x0[i], low[i]), up[i]);
        }
        else {
            if (b[i] > 0)
                x[i] = sol[j];
            else
                x[i] = -sol[j];
            // printf("i = %d, x = %g, b = %g, low = %g, up = %g\n", 
            //     i + 1, x[i], b[i], low[i], up[i]);
            ++j;
        }
   }

    /* Free memory */
    free(sol);
    free_cqk_problem(&prob);

    return res;
}

/*
 * Function Name: newton_projection_2
 *
 * Description: Wrapper for Newton method for projection cont. quad. knapsack 
 *     problem. It should be used by code that wants to pass the vectors that 
 *     describe the problem instead of the cqk_problem structure.
 * 
 *     The problem has the form:
 *
 *     min  1/2 || x - x0 ||_2^2
 *     s.t. b'x = r
 *     low <= x <= up,
 *
 * Input:
 *     int n: dimension of the problem;
 *     double *x0: point to project;
 *     double *b: normal do the hyperplane;
 *     double r: rhs of the hyperplane;
 *     double *low: box lower bound;
 *     double *up: box upper bound.
 * 
 * Output:
 *     double *x: solution obtained.
 * 
 * Return value: the number of iterations required to compute the
 *     solution, -1 in case of failure, -2 in case of infeasible
 *     problem.
 */
int newton_projection_2(int n, double *restrict x0, double *restrict b, 
    double r, double *restrict low, double *restrict up, double *x) {

    /* Copy problem data. */
    cqk_problem prob;
    prob.n = n;
    prob.d = (double *) malloc(n*sizeof(double));
    if (!prob.d) {
        fprintf(stderr, "Memory allocation error, line %d, file %s\n",
                __LINE__, __FILE__);
        exit(1);
    }
    for (unsigned i = 0; i < n; ++i)
        prob.d[i] = 1.0;
    prob.a = x0;
    prob.b = b;
    prob.r = r;
    prob.low = low;
    prob.up = up;

    double *sol = (double *) malloc(n*sizeof(double));
    if (!sol) {
        fprintf(stderr, "Memory allocation error, line %d, file %s\n",
                __LINE__, __FILE__);
        exit(1);
    }

    /* Project */
    int res = newton(&prob, NULL, x);

    /* Free memory */
    free(prob.d);
    prob.d = NULL;

    return res;
}

/*
 * Function Name: newton_cqn
 *
 * Description: Wrapper for Newton method for cont. quad. knapsack
 *     problem. It should be used by code that wants to pass the vectors that
 *     describe the problem instead of the cqk_problem structure.
 *
 *     The problem has the form:
 *
 *     min  1/2 x'Dx - a'x
 *     s.t. b'x = r
 *     low <= x <= up,
 *
 * Input:
 *     int n: dimension of the problem;
 *     double *d: diagonal of D;
 *     double *a: coefficients of the linear term;
 *     double *b: normal do the hyperplane;
 *     double r: rhs of the hyperplane;
 *     double *low: box lower bound;
 *     double *up: box upper bound;
 *     double *x0: initial guess.
 *
 * Output:
 *     double *x: solution obtained.
 *
 * Return value: the number of iterations required to compute the
 *     solution, -1 in case of failure, -2 in case of infeasible
 *     problem.
 */
int newton_cqn(int n, double *restrict d, double *restrict a,
    double *restrict b, double r, double *restrict low, double *restrict up,
    double *x) {

    /* Copy problem data. */
    cqk_problem prob;
    prob.n = n;
    prob.d = d;
    prob.a = a;
    prob.b = b;
    prob.r = r;
    prob.low = low;
    prob.up = up;

    /* Project */
    int res = newton(&prob, NULL, x);

    return res;
}
