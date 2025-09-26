/*
 * Module Name: cont_quad_knapsack
 *
 * Description: Simple interface for the Newton's method for the CQK problem.
 * 
 * Copyright: Paulo J. S. Silva <pjssilva@gmail.com> 2012.
 */

#ifndef NEWTON_PROJECTION_H 
#define NEWTON_PROJECTION_H

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
int newton_projection(int n, double *restrict x0, double *restrict b, double r,
     double *restrict low, double *restrict up, double *x);

#endif