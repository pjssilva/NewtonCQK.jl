/*
 * Module Name: cont_quad_knapsack
 *
 * Description: Implements the semismooth Newton method described in
 *    the manuscript 
 *
 *    R. Cominetti, W. F. Mascarenhas, and Paulo J. S. Silva. "A
 *    Newton method for the continuous quadratic knapsack problem".
 *
 *    It also implements a simple structure to hold the problems.
 *
 * Copyright: Paulo J. S. Silva <pjssilva@gmail.com> 2012.
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "cont_quad_knapsack.h"

/********** cqk_problem functions **********/

/*
 * Function Name: allocate_cqk_problem
 *
 * Description: Allocate the memory necessary hold a CQN problem in
 *     dimension n.
 *
 * Input:
 *     unsigned n: The dimension of the problem.
 * 
 * Output: cqk_problem *p: pointer to the cqk_problem whose memory is
 *     being allocated.
 */
void allocate_cqk_problem(unsigned n, cqk_problem *restrict p) {
    p->n = n;
    p->d = (double *) malloc(p->n*sizeof(double));
    p->a = (double *) malloc(p->n*sizeof(double));
    p->b = (double *) malloc(p->n*sizeof(double));
    p->low = (double *) malloc(p->n*sizeof(double));
    p->up = (double *) malloc(p->n*sizeof(double));
    if (!p->d || !p->a || !p->b || !p->low || !p->up) {
        fprintf(stderr, "Memory allocation error, line %d, file %s\n",
                __LINE__, __FILE__);
        exit(1);
    }
}

/*
 * Function Name: free_cqk_problem
 *
 * Description: Free the memory associated to a cqk_problem.
 *
 * Input/Output: 
 *     cqk_problem *p: at entry a cqk_problem with allocated
 *         memory, at output all the memory will be freed and the pointers
 *         in the cqk_problem structure will point to NULL.
 */
void free_cqk_problem(cqk_problem *restrict p) {
    p->n = 0;
    free(p->d);
    p->d = NULL;
    free(p->a);
    p->a = NULL;
    free(p->b);
    p->b = NULL;
    free(p->low);
    p->low = NULL;
    free(p->up);
    p->up = NULL;
}

/********** Newton method Continuous Quadratic Knapsack problem
            described in the manuscript

    R. Cominetti, W. F. Mascarenhas, and Paulo J. S. Silva. "A Newton
    method for the continuous quadratic knapsack problem".

**********/ 

/*
 * Function Name: initial_multiplier
 *
 * Description: Estimate an initial multiplier from an approximate
 *     solution.
 *
 * Input:
 *     cqk_problem *restrict p: description of the cont. quad. knapsack.
 *     double *restrict x: approximate solution.
 *
 * Output:
 *     double *restrict slopes: slopes of the terms that define the phi 
 *         function.
 *     double *restrict sum_slopes: sum of all slopes.
 *     unsigned *restrict ind: indexes of the active variables that are 
 *         initialized as all. 
 * 
 * Return value: the desired estimate.
 * 
 * Obs: The multiplier is computed using the approximate to estimate
 * the active face obtained from the given solution. It returns the
 * mulplier associated to the problem of projecting into the
 * hyperplane within this face as suggested by the variable fixing
 * methods.
 */
double initial_multiplier(cqk_problem *restrict p, double *restrict x,
                          double *restrict slopes, double *sum_slopes,
                          unsigned *ind) {

    double r = p->r;
    *sum_slopes = 0.0;
    double sum_cts = 0.0;
    double pre_slope;
    /* Estimate the initial multiplier from a given solution. */
    if (x != NULL) {
        for (unsigned i = 0; i < p->n; ++i) {
            ind[i] = i;
            pre_slope = p->b[i]/p->d[i];
            slopes[i] = pre_slope*p->b[i];
            if (x[i] == p->low[i] || x[i] == p->up[i])
                r -= p->b[i]*x[i];
            else {
                sum_cts += pre_slope*p->a[i];
                *sum_slopes += slopes[i];
            }
        }
        /* If all the variables were biding, a very rare and
           unexpected situation, throw the computed values away and
           try again now ignoring all the bounds. */
        if (*sum_slopes == 0.0) {
            r = p->r;
            for(unsigned i = 0; i < p->n; ++i) {
                pre_slope = p->b[i]/p->d[i];
                sum_cts += pre_slope*p->a[i];
                *sum_slopes += pre_slope*p->b[i];
            }
        }     
    }
    else {
        r = p->r;
        for(unsigned i = 0; i < p->n; ++i) {
            ind[i] = i;
            pre_slope = p->b[i]/p->d[i];
            sum_cts += pre_slope*p->a[i];
            slopes[i] = pre_slope*p->b[i];
            *sum_slopes += slopes[i];
        }
    }        

    return (r - sum_cts)/(*sum_slopes);
}

/*
 * Function Name: newton_phi
 *
 * Description: Compute phi(lambda) - r, the function whose zero the
 *     Newton method is trying to compute.
 *
 * Input:
 *     cqk_problem *restrict p: description of the cont. quad. knapsack.
 *     unsigned n: number of free variables.
 *     unsigned *restrict ind: indices of the free variables.
 *     double r: current rhs, after fixing variables.
 *     double slopes: slopes associated to each linear component of phi.
 *     double lambda: current multiplier.      
 *
 * Output:
 *     double *restrict x: problem solution estimated from current multiplier.
 *     double *restrict phi: phi(lamda) - r.
 *     double *restrict deriv: derivative of phi (or an upper bound).
 */
void newton_phi(cqk_problem *restrict p, unsigned n, 
                unsigned *restrict ind, double r, double *restrict slopes, 
                double lambda, double *restrict x, double *restrict phi, 
                double *restrict deriv) {
    *phi = -r;
    double abs_phi = r > 0 ? r : -r;
    *deriv = 0.0;
    for (unsigned i = 0; i < n; ++i) {
        /* Actual index after indirection. */
        unsigned ii = ind[i]; 
        double new_x = (p->a[ii] + lambda*p->b[ii])/p->d[ii];
        if (new_x < p->low[ii])
            new_x = p->low[ii];
        else if (new_x > p->up[ii])
            new_x = p->up[ii];
        else
            /* Note that this will be only an upper bound of the
               correct side derivative if new_x is a breakpoint. */
            *deriv += slopes[ii];
        
        double phi_ii = p->b[ii]*new_x;
        *phi += phi_ii;
        abs_phi += phi_ii > 0 ? phi_ii : -phi_ii;
        x[ii] = new_x;
    }
    DEBUG_PRINT("phi = %e, abs_phi = %e\n", *phi, abs_phi);
    DEBUG_PRINT("deriv = %e\n", *deriv);

    /* Verify if phi was estimated to be sufficiently close to zero and
       zero it to stop the method. */
    if (fabs(*phi) < PREC*abs_phi)
        *phi = 0.0;
}

/*
 * Function Name: breakpoint_to_the_right
 *
 * Description: Find the first positive breakpoint to the right of
 *     lambda.
 *
 * Input:
 *     cqk_problem *restrict p: description of the cont. quad. knapsack.
 *     unsigned n: number of free variables.
 *     unsigned *restrict ind: indices of the free variables.
 *     double lambda: starting point.
 *
 * Return: First positive breakpoint to the right of lambda, if it
 *     exits. Or lambda otherwise.
 */
double breakpoint_to_the_right(cqk_problem *restrict p, unsigned n, 
                               unsigned *restrict ind, double lambda) {
    /* Start at plus infinity */
    double next_break = INVALIDLAMBDA;
    
    for (unsigned i = 0; i < n; ++i) {
        /* Actual index after indirection. */
        unsigned ii = ind[i];
        /* Low (positive) breakpoint associated to the current variable. */
        double pos_break = (p->d[ii]*p->low[ii] - p->a[ii])/p->b[ii];

        if (pos_break > lambda && pos_break < next_break)
            next_break = pos_break;
    }
    return next_break;
}

/*
 * Function Name: breakpoint_to_the_left
 *
 * Description: Find the first negative breakpoint to the left of
 *     lambda.
 *
 * Input:
 *     cqk_problem *restrict p: description of the cont. quad. knapsack.
 *     unsigned n: number of free variables.
 *     unsigned *restrict ind: indices of the free variables.
 *     double lambda: starting point.
 *
 * Return: First negative breakpoint to the left of lambda, if it
 *     exits. Or lambda otherwise.
 */
double breakpoint_to_the_left(cqk_problem *restrict p, unsigned n, 
                              unsigned *restrict ind, double lambda) {
    /* Start at minus infinity */
    double next_break = -INVALIDLAMBDA;
    
    for (unsigned i = 0; i < n; ++i) {
        /* Actual index after indirection. */
        unsigned ii = ind[i];
        /* Up (negative) breakpoint associated to the current variable. */
        double neg_break = (p->d[ii]*p->up[ii] - p->a[ii])/p->b[ii];

        if (neg_break < lambda && neg_break > next_break)
            next_break = neg_break;
    }
    return next_break;
}

/*
 * Function Name: newton_fix
 *
 * Description: Fix variables inside the Newton method.
 *
 * Input:
 *     double phi: phi(lambda) - r.
 *     cqk_problem *restrict p: description of the cont. quad. knapsack.
 *     double *restrict x: primal solution associated to current multiplier.
 *
 * Output:
 *     unsigned *restrict n: number of free variables.
 *     unsigned *restrict ind: new indices of the free variables.
 *     double *restrict r: updated rhs.
 */
void newton_fix(double phi, cqk_problem *restrict p, double *restrict x, 
                unsigned *restrict n, unsigned *restrict ind, 
                double *restrict r) {
    unsigned len = 0;
    if (phi > 0.0)
        for (unsigned i = 0; i < *n; ++i) {
            /* Actual index after indirection. */
            unsigned ii = ind[i];

            if (x[ii] <= p->low[ii]) {
                *r -= p->b[ii]*x[ii];
            }
            else {
                ind[len] = ii;
                ++len;
            }
        }
    else
        for (unsigned i = 0; i < *n; ++i) {
            /* Actual index after indirection. */
            unsigned ii = ind[i];

            if (x[ii] >= p->up[ii]) {
                *r -= p->b[ii]*x[ii];
            }
            else {
                ind[len] = ii;
                ++len;
            }
        }
    DEBUG_PRINT("Fixed %d variables (from %d to %d).\n", 
                *n - len, *n, len);
    *n = len;
}

/*
 * Function Name: secant_step
 *
 * Description: Compute the secant step in the current interval.
 *
 * Input:
 *     bracket *restrict interval: current bracket interval.
 *
 * Return value: the new point after the secant step.
 */
double secant_step(bracket *restrict interval) {
    double secant_point = interval->neg_lambda - interval->negPhi*
        (interval->pos_lambda - interval->neg_lambda) / 
        (interval->posPhi - interval->negPhi);

    /* If the secant step does not decrease the bracket interval, try
       the midpoint. */
    if (secant_point == interval->neg_lambda ||
        secant_point == interval->pos_lambda)
        return 0.5*(interval->neg_lambda + interval->pos_lambda);
    else
        return secant_point;
}

/*
 * Function Name: newton
 *
 * Description: Newton method for the cont. quad. knapsack problem.
 *
 * Input:
 *     cqk_problem *restrict p: the problem description.
 *     double *x0: used it to estimate the initial mulplier if not
 *         NULL.
 * 
 * Output:
 *     double *x: solution obtained.
 * 
 * Return value: the number of iterations required to compute the
 *     solution, -1 in case of failure, -2 in case of infeasible
 *     problem.
 */
int newton(cqk_problem *restrict p, double *x0, double *x) {
    /* Allocate working area */
    unsigned n = p->n;
    unsigned *restrict ind = (unsigned *) malloc(n*sizeof(unsigned));
    double *restrict slopes = (double *) malloc(n*sizeof(double));
    if (!ind || !slopes) {
        fprintf(stderr, "Memory allocation error, line %d, file %s\n",
                __LINE__, __FILE__);
        exit(1);
    }

    /* Initialization */
    unsigned n_iters = 1; /* Number of Newton steps */
    double sum_slopes;    /* Sum of the slopes of the free variables */
    double r = p->r;      /* We want to solve the equation */
    double phi;           /* phi(lambda) - r = 0 */
    double deriv;         /* Derivative of phi */
    double lambda;        /* Multiplier for the affine constraint */
    bracket interval;     /* Interval containing the solution */
    interval.neg_lambda = -INVALIDLAMBDA;
    interval.pos_lambda =  INVALIDLAMBDA;
    
    lambda = initial_multiplier(p, x0, slopes, &sum_slopes, ind);
    newton_phi(p, n, ind, r, slopes, lambda, x, &phi, &deriv);
    DEBUG_PRINT("Initial Lambda=%e, phi - r = %e\n", lambda, phi);

    /* Iteration */
    while (phi != 0.0 && n_iters <= MAXITERS) {
        /* Update bracket interval and verify if it is already too
           small */
        if (phi < 0.0)
            interval.neg_lambda = lambda;
        else
            interval.pos_lambda = lambda;
        if ( (interval.pos_lambda - interval.neg_lambda) < BRACKETPREC*
             MAX(fabs(interval.neg_lambda), fabs(interval.pos_lambda)) ) { 
            phi = 0.0;
            break;
        }
            
        /* Try to fix variables */
        newton_fix(phi, p, x, &n, ind, &r);

        /* Do the Newton step if possible. */
        if (deriv != 0.0) {
            double old_lambda = lambda;
            lambda -= phi/deriv;
            /* If the Newton step is null, the possible precision was
               already achived. */
            if (old_lambda == lambda) {
                phi = 0.0;
                break;
            }

            /* Avoid cycling if necessary */
            if ( interval.neg_lambda > -INVALIDLAMBDA && 
                 interval.pos_lambda < INVALIDLAMBDA &&
                 (lambda >= interval.pos_lambda || 
                  lambda <= interval.neg_lambda) ) {
                DEBUG_PRINT("Possible cycling!\n");
                DEBUG_PRINT("Newton image %e not in (%e, %e)\n", 
                            lambda, interval.neg_lambda, interval.pos_lambda);

                /* Update the values of phi(.) - r at the extremes of the
                   bracket interval. */
                if (phi < 0.0) {
                    interval.negPhi = phi;
                    newton_phi(p, n, ind, r, slopes, interval.pos_lambda, 
                               x, &interval.posPhi, &deriv);
                }
                else {
                    interval.posPhi = phi;
                    newton_phi(p, n, ind, r, slopes, interval.neg_lambda,
                               x, &interval.negPhi, &deriv);
                }
                /* Computing newton_phi is a O(n) operation. Count it
                   as an extra iteration. */
                ++n_iters;

                /* Compute the secant step */
                lambda = secant_step(&interval);

                /* Eliminate at least one break-point */
                double new_bp;
                if (phi < 0.0) {
                    new_bp = breakpoint_to_the_right(p, n, ind, old_lambda);
                    lambda = MAX(lambda, new_bp);
                }
                else {
                    new_bp = breakpoint_to_the_left(p, n, ind, old_lambda);
                    lambda = MIN(lambda, new_bp);
                }
                /* Another O(n) operation */
                ++n_iters;

                DEBUG_PRINT("New lambda: %e\n", lambda);
            }
        }
        else {
            DEBUG_PRINT("Zero derivative at iteration %d\n", n_iters);
            DEBUG_PRINT("Bracket interval [%e, %e]\n", interval.neg_lambda,
                        interval.pos_lambda);

            /* Find the breakpoint close to lambda in the right
               direction. */
            if (phi < 0.0) 
                lambda = breakpoint_to_the_right(p, n, ind, lambda);
            else
                lambda = breakpoint_to_the_left(p, n, ind, lambda);
            DEBUG_PRINT("New lambda: %e\n", lambda);

            /* Since searching for the close breakpoint is a O(n)
               operation, count it as an extra step. */
            ++n_iters;

            /* Test for infeasibility */
            if (lambda <= interval.neg_lambda || 
                lambda >= interval.pos_lambda) {
                interval.pos_lambda = interval.neg_lambda = lambda;
                break;
            }
        }

        /* Compute the function values and derivatives */
        newton_phi(p, n, ind, r, slopes, lambda, x, &phi, &deriv);
        DEBUG_PRINT("Iter %d - lambda = %e, phi - r = %e\n", 
                    n_iters, lambda, phi);
        ++n_iters;
    }

    /* Free working area */
    free(ind);
    free(slopes);
    
    if (phi == 0.0) {
        DEBUG_PRINT("Newton method done!\n\n");
        return n_iters;
    }
    else if (interval.pos_lambda == interval.neg_lambda)
    {
        DEBUG_PRINT("Infeasible problem!\n\n");
        return -2;
    }
    else
    {
        DEBUG_PRINT("Newton method Failed!\n\n");
        return -1;
    }
}
