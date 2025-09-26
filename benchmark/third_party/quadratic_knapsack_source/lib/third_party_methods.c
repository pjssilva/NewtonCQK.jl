/*
 * Module Name: third_party_methods
 *
 * Description: Our implementation of established state-of-the-art
 *     methods to solve the continuous quadratic knapsack problem.
 *
 * Copyright: Paulo J. S. Silva <pjssilva@gmail.com> 2012.
 */

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "third_party_methods.h"

/********** Secant method suggested in the paper

   Dai Y-H, Fletcher R. New algorithms for singly linearly
   constrained quadratic programs subject to lower and upper
   bounds. Mathematical Programming. 2005;106(3):403-421. Available
   at: http://www.springerlink.com/index/10.1007/s10107-005-0595-2.

**********/ 

/*
 * Function Name: secant_initial_multiplier
 *
 * Description: Estimate an initial multiplier from an approximate
 *     solution.
 *
 * Input:
 *     cqk_problem *restrict p: description of the cont. quad. knapsack.
 *     double *restrict x: approximate solution.
 *
 * Return value: the desired estimate.
 * 
 * Obs: The multiplier is computed using the approximate to estimate
 * the active face obtained from the given solution. It returns the
 * mulplier associated to the problem of projecting into the
 * hyperplane within this face as suggested by the variable fixing
 * methods.
 *
 * Obs: We decided to re-implement this method instead of using the 
 *     original version from cont_quad_knapsack.c because the secant
 *     method does not use variable fixing and does not need the slopes
 *     of the phi function.
 */
double secant_initial_multiplier(cqk_problem *restrict p, double *restrict x) {
    /* Estimate a initial multiplier if possible. */
    double r = p->r;
    double sum_slopes = 0.0;
    double sum_cts = 0.0;
    double pre_slope;
    if (x != NULL)
        for (unsigned i = 0; i < p->n; ++i) {
            if (x[i] == p->low[i] || x[i] == p->up[i])
                r -= p->b[i]*x[i];
            else {
                pre_slope = p->b[i]/p->d[i];
                sum_cts += pre_slope*p->a[i];
                sum_slopes += pre_slope*p->b[i];
            }
        }

    /* If the previous solution was not given or if all the variables
       were biding, a very rare and unexpected situation, throw the
       computed values away and try again now ignoring all the
       bounds. */
    if (sum_slopes == 0.0) {
        r = p->r;
        for(unsigned i = 0; i < p->n; ++i) {
            pre_slope = p->b[i]/p->d[i];
            sum_cts += pre_slope*p->a[i];
            sum_slopes += pre_slope*p->b[i];
        }
    }        
    return (r - sum_cts)/sum_slopes;
}

/*
 * Function Name: secant_phi
 *
 * Description: Compute phi(lambda) - r, the function whose zero the
 *    secant method is trying to compute, and the (primal) solution
 *    associated to lambda.
 *
 * Input:
 *    cqk_problem *restrict p: description of the cont. quad. knapsack.
 *    double lambda: current multiplier.      
 *
 * Output: 
 *    double *restrict x: probblem solution estimated from current
 *       multiplier.
 *
 * Return value: phi(lambda) - r
 */
double secant_phi(cqk_problem *restrict p, double lambda, 
                  double *restrict x) {

    double phi = -p->r;
    double abs_phi = p->r > 0 ? p->r : -p->r;
    for (unsigned i = 0; i < p->n; ++i) {
        double new_x;

        new_x = (p->a[i] + lambda*p->b[i])/p->d[i];
        if (new_x < p->low[i])
            new_x = p->low[i];
        else if (new_x > p->up[i])
            new_x = p->up[i];
        
        phi += p->b[i]*new_x;
        abs_phi += new_x > 0 ? p->b[i]*new_x : -p->b[i]*new_x;
        x[i] = new_x;
    }
    
    DEBUG_PRINT("phi = %.16e, abs_phi = %e\n", phi, abs_phi);
    /* Verify if phi was estimated to be sufficiently close to zero and
       zero it to end projection. */
    if (fabs(phi) < PREC*abs_phi)
        phi = 0.0;
    return phi;
}

/*
 * Function Name: bracketing
 *
 * Description: Find a bracketing interval containing the solution of
 *    phi(lambda) = r. This is Algorithm 1 in the paper of Dai and
 *    Fletcher.
 *
 * Input:
 *    cqk_problem *restrict p: description of the cont. quad. knapsack.
 *    double lambda: initial guess.
 *    double delta_lambda: initial step size.
 *
 * Output:
 *    double *restrict x: problem solution vector estimated from
 *       last multiplier.
 *    bracket *restrict interval: Desired bracketing interval.
 *
 * Return value: number of calls to secant_phi (iterations).
 */
unsigned bracketing(cqk_problem *restrict p, 
                    double lambda, double delta_lambda, 
                    double *restrict x, bracket *restrict interval)  {

    unsigned n_iter = 0;
    double phi = secant_phi(p, lambda, x);
    ++n_iter;
    if (phi == 0.0) {
        interval->neg_lambda = interval->pos_lambda = lambda;
        interval->negPhi = interval->posPhi = 0.0;
        return n_iter;
    }
    else if (phi < 0.0) {
        double 
            lambda_low = lambda,
            phi_low = phi;
        lambda += delta_lambda;
        phi = secant_phi(p, lambda, x);
        ++n_iter;
        while (phi < 0.0) {
            double s = MAX(phi_low/phi - 1.0, 0.1);
            lambda_low = lambda;
            phi_low = phi;
            delta_lambda += delta_lambda/s;
            lambda += delta_lambda;
            phi = secant_phi(p, lambda, x);
            ++n_iter;
        }
        interval->neg_lambda = lambda_low;
        interval->negPhi = phi_low;
        interval->pos_lambda = lambda;
        interval->posPhi = phi;
    }
    else {
        double 
            lambda_up = lambda,
            phi_up = phi;
        lambda -= delta_lambda;
        phi = secant_phi(p, lambda, x);
        ++n_iter;
        while (phi > 0.0) {
            double s = MAX(phi_up/phi - 1.0, 0.1);
            lambda_up = lambda;
            phi_up = phi;
            delta_lambda += delta_lambda/s;
            lambda -= delta_lambda;
            phi = secant_phi(p, lambda, x);
            ++n_iter;
        }
        interval->neg_lambda = lambda;
        interval->negPhi = phi;
        interval->pos_lambda = lambda_up;
        interval->posPhi = phi_up; 
    }
    return n_iter;
}

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
           double *restrict deltaLambda, double *x) {

    double lambda0 = secant_initial_multiplier(prob, x0);
    unsigned n_iter = 1;
        
    if (*deltaLambda < 0.0)
        *deltaLambda = 2.0;

    /* Geta bracket interval */
    bracket interval;
    n_iter += bracketing(prob, lambda0, *deltaLambda, x, &interval);

    /* Verify if a solution was found in the bracketing phase */
    if (interval.negPhi == 0.0 ) {
        *deltaLambda = 1.0 + fabs(interval.neg_lambda - lambda0);
        return n_iter;
    }
    if (interval.posPhi == 0.0) {
        *deltaLambda = 1.0 + fabs(interval.pos_lambda - lambda0);
        return n_iter;
    }
        
    /* Start secant phase */
    double s = 1.0 - interval.negPhi/interval.posPhi;
    double deltaL = (interval.pos_lambda - interval.neg_lambda)/s;
    double lambda = interval.pos_lambda - deltaL;
    double phi = secant_phi(prob, lambda, x);
    ++n_iter;
    while (phi != 0.0 && n_iter <= MAXITERS) {
        /* If the bracket interval is already too small, stop. */
        if ( (interval.pos_lambda - interval.neg_lambda) < BRACKETPREC*
             MAX(fabs(interval.neg_lambda), fabs(interval.pos_lambda)) ) { 
            phi = 0.0;
            break;
        }

        if (phi > 0.0) {
            if (s <= 2.0) {
                interval.pos_lambda = lambda;
                interval.posPhi = phi;
                s = 1.0 - interval.negPhi/interval.posPhi;
                deltaL = (interval.pos_lambda - interval.neg_lambda)/s;
                lambda = interval.pos_lambda - deltaL;
            }
            else {
                s = MAX(interval.posPhi/phi - 1.0, 0.1);
                deltaL = (interval.pos_lambda - lambda)/s;
                double lambda_new = MAX(lambda - deltaL, 
                                        0.75*interval.neg_lambda + 0.25*lambda);
                interval.pos_lambda = lambda;
                interval.posPhi = phi;
                lambda = lambda_new;
                s = (interval.pos_lambda - interval.neg_lambda) / 
                    (interval.pos_lambda - lambda);
            }
        }
        else {
            if (s >= 2.0) {
                interval.neg_lambda = lambda;
                interval.negPhi = phi;
                s = 1.0 - interval.negPhi/interval.posPhi;
                deltaL = (interval.pos_lambda - interval.neg_lambda)/s;
                lambda = interval.pos_lambda - deltaL;
            }
            else {
                s = MAX(interval.negPhi/phi - 1.0, 0.1);
                deltaL = (lambda - interval.neg_lambda)/s;
                double lambda_new = MIN(lambda + deltaL,
                                        0.75*interval.pos_lambda + 0.25*lambda);
                interval.neg_lambda = lambda;
                interval.negPhi = phi;
                lambda = lambda_new;
                s = (interval.pos_lambda - interval.neg_lambda) / 
                    (interval.pos_lambda - lambda);
            }
        }
        phi = secant_phi(prob, lambda, x);
        ++n_iter;
    }
    *deltaLambda = 1.0 + fabs(lambda0 - lambda);
    if (phi == 0.0) {
        DEBUG_PRINT("Secant method done!\n\n");
        return n_iter;
    }
    else {
        DEBUG_PRINT("Secant method Failed!\n\n");
        return -1;
    }
}

/********** Variable fixing method (variation of the Britan and Harx
            method) suggested in the paper

   Kiwiel KC. Variable Fixing Algorithms for the Continuous
   Quadratic Knapsack Problem. Journal of Optimization Theory and
   Applications. 2008;136(3):445-458. Available at:
   http://www.springerlink.com/index/10.1007/s10957-007-9317-7.

**********/ 

/*
 * Function Name: feasibility
 *
 * Description: Compute the feasibility measure associated to lambda,
 *    as described in Kiwiel's paper. It also finds the primal
 *    solution associated to the multiplier.
 *
 * Input:
 *    cqk_problem *restrict p: description of the cont. quad. knapsack.
 *    double slopes: slopes associated to each linear component of phi.
 *    double lambda: current multiplier.      
 *    double r: current rhs, after fixing variables.
 *    unsigned n: number of free variables.
 *    unsigned *restrict ind: indices of the free variables.
 *
 * Output:
 *    double *restrict nabla: negative component of the feasibility.
 *    double *restrict delta: positive component of the feasibility.
 *    double *restrict x: problem solution estimated from current 
 *       multiplier.
 */
void feasibility(cqk_problem *restrict p, double *restrict slopes, 
                 double lambda, unsigned n, unsigned *restrict ind, 
                 double *restrict x, double *restrict nabla, 
                 double *restrict delta) {

    *nabla = 0.0;
    *delta = 0.0;
    for (unsigned i = 0; i < n; ++i) {
        /* Actual index after indirection. */
        unsigned ii = ind[i];

        double new_x = (p->a[ii] + lambda*p->b[ii])/p->d[ii];
        if (new_x <= p->low[ii]) {
            *nabla += p->b[ii]*(p->low[ii] - new_x);
            new_x = p->low[ii];
        } else if (new_x >= p->up[ii]) {
            *delta += p->b[ii]*(new_x - p->up[ii]);
            new_x = p->up[ii];
        }
        x[ii] = new_x;
    }

    if (fabs(*nabla - *delta) <= PREC*(*nabla + *delta)) {
        *nabla = *delta;
    }
    DEBUG_PRINT("Feasibility = %.16e\n", *nabla - *delta);
}

/*
 * Function Name: kiwiel_fix
 *
 * Description: Fix variables whenever possible.
 *
 * Input:
 *    double feas: feasibility measure.
 *    cqk_problem *restrict p: description of the cont. quad. knapsack.
 *    double *restrict slopes: slopes associated to each linear component 
 *       of phi.
 *    double *restrict x: primal solution associated to current multiplier.
 *
 * Output:
 *    double *restrict r: updated rhs.
 *    unsigned *restrict n: number of free variables.
 *    unsigned *restrict ind: new indices of the free variables.
 *    double *restrict q: new sum of slopes.
 */
void kiwiel_fix(double feas, cqk_problem *restrict p, double *restrict slopes, 
                double *restrict x, double *restrict r, unsigned *restrict n, 
                unsigned *restrict ind, double *restrict q) {

    unsigned len = 0;
    if (feas > 0)
        for (unsigned i = 0; i < *n; ++i) {
            /* Actual index after indirection. */
            unsigned ii = ind[i];

            if (x[ii] <= p->low[ii]) {
                *q -= slopes[ii];
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
                *q -= slopes[ii];
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
int kiwiel_var_fix(cqk_problem *restrict p, double *restrict x) {
    /* Allocate and initialize working area*/
    unsigned n = p->n;
    unsigned *restrict ind = (unsigned *) malloc(n*sizeof(unsigned));
    double *restrict slopes = (double *) malloc(n*sizeof(double));
    if (!ind || !slopes) {
        fprintf(stderr, "Memory allocation error, line %d, file %s\n",
                __LINE__, __FILE__);
        exit(1);
    }

    /* Initialization */
    unsigned n_iters = 0; /* Number of iterations */
    double 
        nabla = 0.0,      /* We want to solve the equation */
        delta = 1.0,      /* nabla = delta; */
        r = p->r,         /* Current rhs of the linear constraint */
        q,                /* Sum of the slopes of the free variables */
        lambda;           /* Multiplier with simple initial value */

    /* Estimate initial multiplier and initialize the free variable
       list. */
    DEBUG_PRINT("Starting Kiwiel variable fixing method\n");
    lambda = initial_multiplier(p, NULL, slopes, &q, ind);

    /* Start iteration */
    feasibility(p, slopes, lambda, n, ind, x, &nabla, &delta);
    n_iters++;
    DEBUG_PRINT("Initial lambda = %e, feasibility = %e\n",
                lambda, nabla - delta);
    while(nabla != delta) {
        if (n_iters > MAXITERS) break;
        
        kiwiel_fix(nabla - delta, p, slopes, x, &r, &n, ind, &q);
        if (nabla > delta)
            lambda -= nabla/q;
        else
            lambda += delta/q;

        feasibility(p, slopes, lambda, n, ind, x, &nabla, &delta);
        DEBUG_PRINT("iter %d - lambda = %e, feasibility = %e\n", 
                    n_iters, lambda, nabla - delta);
        ++n_iters;
    }

    /* Free working memory */
    free(ind);
    free(slopes);

    /* Return the number of iterations */
    if (nabla == delta) {
        DEBUG_PRINT("Kiwiel projection done!\n\n");
        return n_iters;
    }
    else {
        DEBUG_PRINT("Kiwiel projection failed!\n\n");
        return -1;
    }
}

/********** Median search method using Kiwiel's SELECT implementation
            suggested in the paper

   Kiwiel KC. Breakpoint searching algorithms for the continuous
   quadratic knapsack problem. Mathematical
   Programming. 2008;112(2):473-491. Available at:
   http://www.springerlink.com/index/10.1007/s10107-006-0050-z.

**********/ 

/*  Declare Kiwiel's implementation of SELECT algorithm. */
void dsel05_(int *restrict k, int *restrict len, double *restrict x);

/*
 * Function Name: median
 *
 * Description: Wrapper to call Kiwiel's SELECT implementation to find
 *    the median of a vector x.
 *
 * Input:
 *    unsigned n: dimension of the vector.
 * 
 * Input/Output: 
 *    double *restrict x: original vector at input and the vector
 *    with the elements smaller than the median at the begining at
 *    output.
 *
 * Return value: the median
 */
double median(unsigned n, double *restrict x) {
    int k = (int) n/2 + 1;
    int len = (int) n;
    dsel05_(&k, &len, x);
    return x[k - 1];
}

/*
 * Function Name: median_phi
 *
 * Description: Compute phi(t) using intermediate values maintained by
 *    the median fiding algorithm.
 *
 * Input:
 *    cqk_problem *prob: description of the cont. quad. knapsack.
 *    unsigned *restrict I: indices of the free variables.
 *    unsigned Ilen: number of free variables.
 *    double p: itermediate value computed by the median search method.
 *    double q: itermediate value computed by the median search method.
 *    double s: itermediate value computed by the median search method.
 *    double *restrict Tl: list of 'low' breakpoints.
 *    double *restrict Tu: list of 'up' breakpoints.
 *    double t: current multiplier.      
 *
 * Return value: phi(t)
 */
double median_phi(cqk_problem *restrict prob, unsigned *restrict I, 
                  unsigned Ilen, double p, double q, double s, 
                  double *restrict Tl, double *restrict Tu, double t) {
    /* compute phi of th */
    double phi = (p - t*q) + s;
    for (unsigned i = 0; i < Ilen; ++i) {
        double tli = Tl[I[i]];
        double tui = Tu[I[i]];
        if (t >= tui && t <= tli)
            phi += prob->b[I[i]]*(prob->a[I[i]] 
                                  - t*prob->b[I[i]])/prob->d[I[i]];
        else if (tli < t)
            phi += prob->b[I[i]]*prob->low[I[i]];
        else
            phi += prob->b[I[i]]*prob->up[I[i]];
    }
    return phi;
}

/*
 * Function Name: kiwiel_search
 *
 * Description: Kiwiel's variation of the median search method for
 *    the cont. quad. knapsack problem. The method is based on 
 *
 *    Kiwiel KC. Breakpoint searching algorithms for the continuous
 *    quadratic knapsack problem. Mathematical
 *    Programming. 2008;112(2):473-491. Available at:
 *    http://www.springerlink.com/index/10.1007/s10107-006-0050-z.
 *
 * Input:
 *    cqk_problem *restrict p: the problem description.
 * 
 * Output:
 *    double *restrict x: solution vector.
 * 
 * Return value: the number of iterations required to compute the
 *    solution or -1 in case of failure.
 */
int kiwiel_search(cqk_problem *restrict prob, double *restrict x) {
    
    /* Allocate working area */
    unsigned n = prob->n;
    unsigned *restrict I = (unsigned *) malloc(n*sizeof(unsigned));
    double *restrict Tl = (double *) malloc(4*n*sizeof(double));
    double *restrict Tu = Tl + n;
    double *restrict T = Tu + n;
    if (!I || !Tl) {
        fprintf(stderr, "Memory allocation error, line %d, file %s\n",
                __LINE__, __FILE__);
        exit(1);
    }

    /* Initialization */
    for (unsigned i = 0; i < n; ++i) {
        I[i] = i;
        Tl[i] = (prob->a[i] - prob->low[i]*prob->d[i])/prob->b[i];
        Tu[i] = (prob->a[i] - prob->up[i]*prob->d[i])/prob->b[i];
        T[2*i] = Tu[i];
        T[2*i + 1] = Tl[i];
    }

    unsigned 
        Ilen = n,
        Tlen = 2*n;
    double 
        tL = -INVALIDLAMBDA,
        tU = +INVALIDLAMBDA,
        p = 0.0,
        q = 0.0,
        s = 0.0;
    unsigned n_iters = 0;

    /* Iteration */
    double phi, th;
    while (1) {
        ++n_iters;
        th = median(Tlen, T);

        /* Compute phi of th */
        phi = median_phi(prob, I, Ilen, p, q, s, Tl, Tu, th);

        /* Optimality check */
        if (phi == prob->r) break;
            
        /* Lower breakpoint removal */
        if (phi > prob->r) {
            tL = th;
            T = T + Tlen/2 + 1;
            Tlen -= Tlen/2 + 1;
        }
        /* Upper breakpoint removal */
        else {
            tU = th;
            Tlen = Tlen/2;
        }

        /* Updating I, p, q, s to compute the next phi efficiently. */
        unsigned j = 0;
        for (unsigned i = 0; i < Ilen; ++i) {
            double tli = Tl[I[i]];
            double tui = Tu[I[i]];
            if (tli <= tL)
                s += prob->b[I[i]]*prob->low[I[i]];
            else if (tU <= tui)
                s += prob->b[I[i]]*prob->up[I[i]];
            else if (tui <= tL && tL <= tli && tui <= tU && tU <= tli) {
                p += prob->a[I[i]]*prob->b[I[i]]/prob->d[I[i]];
                q += prob->b[I[i]]*prob->b[I[i]]/prob->d[I[i]];
            }
            else {
                I[j] = I[i];
                ++j;
            }
        }
        Ilen = j;

        /* Stopping criterion. */
        if (Tlen == 0) {
            if (tL < tU) {
                double gtL = (p - tL*q) + s;
                double gtU = (p - tU*q) + s;
                th = tL - (gtL - prob->r)*(tU - tL)/(gtU - gtL);
            }
            else
                th = tL;
            break;
        }
    }
    
    /* Compute the solution using the optimal multiplier */
    for (unsigned i = 0; i < n; ++i) {
        x[i] = (prob->a[i] - th*prob->b[i])/prob->d[i];
        if (x[i] < prob->low[i])
            x[i] = prob->low[i];
        else if (x[i] > prob->up[i])
            x[i] = prob->up[i];
    }

    /* Free working memory */
    free(I);
    free(Tl);
 
    return n_iters;
}
