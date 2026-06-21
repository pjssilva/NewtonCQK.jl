// Based on the file run_proj.c from the PPROJ package

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include "pproj_user.h"

/*
 * Project y onto the unit simplex
 * We need to construct the constraint matrix and lower/upper bounds vectors
 * on the variables. As these objects are constant when solving several CQK's in
 * sequence, we pass them as parameters. All we need is
 *   Ap: columns pointers
 *   Ai: row indices
 *   ones: vector of 1's
 *   zeros: vector of 0's
 * We have Ap[i] = i and Ai[i] = 0 for all 0 <= i <= n-1 and Ap[n] = n.
 */
int pproj_proj(int n, double *restrict y, int *Ap, int *Ai, double *zeros,
    double *ones, double *x, double grad_tol)
{
    int status = -1;
    double lambda[1] = {0.0};
    double one[1] = {1.0};
    PPparm Parm;

    pproj_default (&Parm) ;
    // Parm.cholmod = 0;
    // Parm.multilevel = 0;
    // Parm.use_lambda = 0;
    Parm.ScaleSigma = 0;
    // Parm.stopabs = 1;

    status = pproj (x, lambda, NULL,
                    grad_tol,       // grad_tol
                    &Parm,          // default parameters
                    NULL,
                    y,              // point to be projected
                    1, n,           // nrow, ncol
                    Ap,             // columns pointers
                    Ai,             // row indices
                    ones,           // constraints vals
                    zeros, ones,    // low, up (bounds on variables)
                    one, one        // L, U (bounds on constraints)
                    ) ;

    if ( status != 0 ) status = -1;

    return status;
}

/*
 * Solve general CQK
 * We need to scaling the problem to put it in the format accepted by PPROJ.
 *
 * CQK
 * min_y 0.5 xt D x - at x   s.t.   bt x = r,  l <= x <= u
 *
 * Writing z = D^{1/2}x, we can rewrite the problem as
 *
 * min_z 0.5 * |z - D^{-1/2}a|^2
 * s.t.  (D^{-1/2}b)t z = r,  D^{1/2}l <= z <= D^{1/2}u
 *
 * In the function pproj_cqk, we have to pass the scaled vectors
 * sc_a = D^{-1/2}a, sc_b = D^{-1/2}b, sc_l = D^{1/2}l and sc_u = D^{1/2}u
 * instead of a, b, low and up. This because these vectors can be reused to
 * solve a sequence of CQKs. However, note that the final solution z must be
 * scaled to obtain the original x, so this cost must be considered.
 */

int pproj_cqk(int n, double *restrict d, double *restrict sc_a,
    double *restrict sc_b, double r, double *restrict sc_low,
    double *restrict sc_up, int *restrict Ap, int *restrict Ai, double *x, double grad_tol)
{
    int i;
    int status = -1;
    double lambda[1] = {0.0};
    double rhs[1] = {r};
    PPparm Parm;

    // for (i = 0; i <= n; i++) Ap[i] = i;

    pproj_default (&Parm) ;
    // Parm.cholmod = 0;
    // Parm.multilevel = 0;
    // Parm.use_lambda = 0;
    Parm.ScaleSigma = 0;
    // Parm.stopabs = 1;

    status = pproj (x, lambda, NULL,
                    grad_tol,       // grad_tol
                    &Parm,          // default parameters
                    NULL,
                    sc_a,           // point to be projected
                    1, n,           // nrow, ncol
                    Ap,             // columns pointers
                    Ai,             // row indices
                    sc_b,           // constraints vals
                    sc_low, sc_up,  // low, up (bounds on variables)
                    rhs, rhs        // L, U (bounds on constraints)
                    ) ;

    if ( status != 0 )
    {
        status = -1;
        goto TERMINATE;
    }

    // scale solution
    for (i = 0; i < n; i++)
        x[i] /= sqrt(d[i]);

    TERMINATE:

    return status;
}