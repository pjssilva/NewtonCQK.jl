// Based on the file run_proj.c from the PPROJ package

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include "pproj_user.h"

// Project y onto the unit simplex
int pproj_proj(int n, double *restrict y, double *x, double grad_tol)
{
    int i;
    int status = -1;
    double lambda[1] = {0.0};
    double one[1] = {1.0};
    double *zeros, *ones;
    int *Ap, *Ai ;

    Ap = (int *) malloc ((n+1)*sizeof (int)) ;
    if (!Ap) goto TERMINATE;
    Ai = (int *) malloc (n*sizeof (int)) ;
    if (!Ai) goto TERMINATE;
    ones = (double *) malloc (n*sizeof (double)) ;
    if (!ones) goto TERMINATE;
    zeros = (double *) malloc (n*sizeof (double)) ;
    if (!zeros) goto TERMINATE;

    for (i = 0; i < n; i++)
    {
        zeros[i] = 0.0;
        ones[i] = 1.0;
        Ap[i] = i;
        Ai[i] = 0;
    }
    Ap[n] = n;

    status = pproj (x, lambda, NULL,
                    grad_tol,       // grad_tol
                    NULL,           // default parameters
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

    TERMINATE:

    if (Ap) free (Ap) ;
    if (Ai) free (Ai) ;
    if (zeros) free (zeros);
    if (ones) free (ones);

    return status;
}

// Solve general CQK
// int pproj_cqk(int n, double *restrict d, double *restrict a,
//     double *restrict b, double r, double *restrict low, double *restrict up,
//     double *x, double grad_tol)
// {
//     INT i, j, p, ni, anz, status ;
//     double t, errb, errx, absAx, grad_tol, err, normx,
//           *bu, *bl, *Ax, *lo, *hi, *lambda, *y, *work ;
//     INT *Ap, *Ai ;
//     PPparm Parm ;
//
//     pproj_default (&Parm) ;
//
//     bu = (double *) malloc (nrow*sizeof (double)) ;
//     bl = (double *) malloc (nrow*sizeof (double)) ;
//     Ap = (int *) malloc ((ncol+1)*sizeof (int)) ;
//     Ai = (int *) malloc (anz*sizeof (int)) ;
//     Ax = (double *) malloc (anz*sizeof (double)) ;
//     lo = (double *) malloc (ncol*sizeof (double)) ;
//     hi = (double *) malloc (ncol*sizeof (double)) ;
//     lambda = (double *) malloc (nrow*sizeof (double)) ;
//     y = (double *) malloc (ncol*sizeof (double)) ;
//     for (j = 0; j <= ncol; j++)
//     {
//         fscanf (f, "%d\n",  Ap+j) ;
//     }
//     for (p = 0; p < anz ; p++)
//     {
//         fscanf (f, "%d\n",  Ai+p) ;
//     }
//     for (p = 0; p < anz ; p++)
//     {
//         fscanf (f, "%lg\n", Ax+p) ;
//         if ( Ax [p] == 0 )
//         {
//             printf ("error, Ax [%d] == 0 on input", p) ;
//             exit (0) ;
//         }
//     }
//     for (j = 0; j < ncol ; j++)
//     {
//         fscanf (f, "%lg %lg\n", lo+j, hi+j) ;
//     }
//     for (i = 0; i < nrow ; i++)
//     {
//         fscanf (f, "%lg %lg\n", bl+i, bu+i) ;
//     }
//     for (j = 0; j < ncol ; j++)
//     {
//         fscanf (f, "%lg\n", y+j) ;
//     }
//
//     ni = 0 ;
//     for (i = 0; i < nrow; i++)
//     {
//         if ( bl [i] < bu [i] ) ni++ ;
//     }
//
//     grad_tol = 1.e-9 ;
//     /* set nondefault parameter values in the Parm structure
//         before calling pproj */
//
//     status = pproj (x, lambda, NULL, grad_tol, &Parm, NULL, y, nrow, ncol,
//                     Ap, Ai, Ax, lo, hi, bl, bu) ;
//
//     if ( !Parm.stopabs && (absAx != 0.) )
//     {
//         // If parm->stopabs = TRUE, then the stopping criterion is the absolute condition
//         // ||grad L (lambda)||_sup <= grad_tol
//     }
//
//     if ( status != 0 )
//     {
//         // problem not solved
//     }
//
//     free (bu) ;
//     free (bl) ;
//     free (Ap) ;
//     free (Ai) ;
//     free (Ax) ;
//     free (lo) ;
//     free (hi) ;
//     free (lambda) ;
//     free (y) ;
//
//     fclose (namefile) ;
//     return 0;
// }
