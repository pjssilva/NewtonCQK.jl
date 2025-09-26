/*
 * Module Name: third_party_methods
 *
 * Description: Our implementation of established state-of-the-art
 *     methods to solve the continuous quadratic knapsack problem.
 *
 * Copyright: Paulo J. S. Silva <pjssilva@gmail.com> 2012.
 */

#ifndef THIRD_PARTY_METHODS_H
#define THIRD_PARTY_METHODS_H

#include "cont_quad_knapsack.h"

/*
 * Function Name: secant
 *
 * Description: Secant method for the cont. quad. knapsack
 *     problem. The method is based on 
 *
 *     Dai Y-H, Fletcher R. New algorithms for singly linearly
 *     constrained quadratic programs subject to lower and upper
 *     bounds. Mathematical Programming. 2005;106(3):403-421. Available
 *     at: http://www.springerlink.com/index/10.1007/s10107-005-0595-2.
 *
 * Input:
 *     cqk_problem *p: the problem description
 *     double *x0: If non NULL, try to use it to estimate the initial 
 *         multiplier. 
 *
 * Input/Output:
 *     double *restrict deltaLambda: initial stepsize for the
 *         bracketing phase. If negative a default value of 2.0 will be
 *         used. At output it will recieve the stepsize required to move
 *         from lambda0 to the final multiplier + 1.0 (as suggested in
 *         Dai and Fletcher paper).
 * 
 * Output:
 *     double *restrict x: solution vector.
 * 
 * Return value: the number of iterations required to compute the
 *     solution or -1 in case of failure.
 */
int secant(cqk_problem *restrict prob, double *x0,
           double *restrict deltaLambda, double *x);

/*
 * Function Name: kiwiel_var_fix
 *
 * Description: Kiwiel's variation of the variable fixing method for
 *     the cont. quad. knapsack problem. The method is based on 
 *
 *     Kiwiel KC. Variable Fixing Algorithms for the Continuous
 *     Quadratic Knapsack Problem. Journal of Optimization Theory and
 *     Applications. 2008;136(3):445-458. Available at:
 *     http://www.springerlink.com/index/10.1007/s10957-007-9317-7.
 *
 * Input:
 *     cqk_problem *restrict p: the problem description.
 * 
 * Output:
 *     double *restrict x: solution vector.
 * 
 * Return value: the number of iterations required to compute the
 *     solution or -1 in case of failure.
 */
int kiwiel_var_fix(cqk_problem *restrict p, double *restrict x);

/*
 * Function Name: kiwiel_search
 *
 * Description: Kiwiel's variation of the median search method for
 *     the cont. quad. knapsack problem. The method is based on 
 *
 *     Kiwiel KC. Breakpoint searching algorithms for the continuous
 *     quadratic knapsack problem. Mathematical
 *     Programming. 2008;112(2):473-491. Available at:
 *     http://www.springerlink.com/index/10.1007/s10107-006-0050-z.
 *
 * Input:
 *     cqk_problem *restrict p: the problem description.
 * 
 * Output:
 *     double *restrict x: solution vector
 * 
 * Return value: the number of iterations required to compute the
 *     solution or -1 in case of failure.
 */
int kiwiel_search(cqk_problem *restrict prob, double *restrict x);

#endif
