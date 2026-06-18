/* ========================================================================== */
/* Read in a polyhedron in the form:

                 bl <= Ax <= bu,  lo <= x <= hi

   Scale the rows and columns of A.  Generate a random point and
   project it onto the polyhedron. */
/* ========================================================================== */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include "pproj_user.h"
#define MAXCHAR 3000
#define MAXLINE 256

double pproj_timer ( void ) ;

void pproj_error
(
    int status,
    const char *file,
    int line,
    const char *message
) ;

int main (void /*int argc, char **argv*/)
{
    char s [MAXLINE+1], *testprobs, *probname, fullpathname [MAXCHAR] ;
    INT i, j, p, nrow, ncol, ni, anz, status, probnum ;
    double t, errb, errx, absAx, tic, elapsed, grad_tol, err, normx,
          *bu, *bl, *Ax, *lo, *hi, *lambda, *x, *y, *work ;
    FILE *f, *namefile ;
    INT *Ap, *Ai ;
    PPparm Parm ;
    PPstat Stat ;

    testprobs = "/home/data/Probs/Polyhedral/" ;
    namefile = fopen ("names", "r") ;
    pproj_default (&Parm) ;
    /* ---------------------------------------------------------------------- */
    /* read LPDASA problem names */
    /* ---------------------------------------------------------------------- */

    probnum = 0 ;
    while (fgets (s, MAXLINE, namefile) != (char *) NULL)
    {
        probnum++ ;
        for (probname = s; *probname; probname++)
        {
            if (isspace (*probname)) *probname = '\0' ;
        }
        probname = s ;
        strcpy (fullpathname, testprobs) ;
        strcat (fullpathname, probname) ;
        /* printf ("reading problem: %s\n", fullpathname) ;*/
        f = fopen (fullpathname, "r") ;
        if (f == (FILE *) NULL)
        {
            printf ("file not found\n") ;
            exit (0) ;
        }

        fscanf (f, "%i %i %i\n", &nrow, &ncol, &anz) ;
        bu = (double *) malloc (nrow*sizeof (double)) ;
        bl = (double *) malloc (nrow*sizeof (double)) ;
        Ap = (int *) malloc ((ncol+1)*sizeof (int)) ;
        Ai = (int *) malloc (anz*sizeof (int)) ;
        Ax = (double *) malloc (anz*sizeof (double)) ;
        lo = (double *) malloc (ncol*sizeof (double)) ;
        hi = (double *) malloc (ncol*sizeof (double)) ;
        lambda = (double *) malloc (nrow*sizeof (double)) ;
        x = (double *) malloc (ncol*sizeof (double)) ;
        y = (double *) malloc (ncol*sizeof (double)) ;
        for (j = 0; j <= ncol; j++)
        {
            fscanf (f, "%d\n",  Ap+j) ;
        }
        for (p = 0; p < anz ; p++)
        {
            fscanf (f, "%d\n",  Ai+p) ;
        }
        for (p = 0; p < anz ; p++)
        {
            fscanf (f, "%lg\n", Ax+p) ;
            if ( Ax [p] == 0 )
            {
                printf ("error, Ax [%d] == 0 on input", p) ;
                exit (0) ;
            }
        }
        for (j = 0; j < ncol ; j++)
        {
            fscanf (f, "%lg %lg\n", lo+j, hi+j) ;
        }
        for (i = 0; i < nrow ; i++)
        {
            fscanf (f, "%lg %lg\n", bl+i, bu+i) ;
        }
        fclose (f) ;

        /* read the infeasible point */
        strcpy (fullpathname, testprobs) ;
        strcat (fullpathname, "InfeasiblePoints/");
        strcat (fullpathname, probname) ;
        /* printf ("reading point: %s\n", fullpathname) ;*/
        f = fopen (fullpathname, "r") ;
        if (f == (FILE *) NULL)
        {
            printf ("file not found\n") ;
            exit (0) ;
        }
        for (j = 0; j < ncol ; j++)
        {
            fscanf (f, "%lg\n", y+j) ;
        }
        fclose (f) ; 

        ni = 0 ;
        for (i = 0; i < nrow; i++)
        {
            if ( bl [i] < bu [i] ) ni++ ;
        }

        grad_tol = 1.e-9 ;
        /* set nondefault parameter values in the Parm structure
           before calling pproj */
        tic = pproj_timer () ;
        status = pproj (x, lambda, NULL, grad_tol, &Parm, &Stat, y, nrow, ncol,
                        Ap, Ai, Ax, lo, hi, bl, bu) ;
        elapsed = pproj_timer () - tic ;

        /* evaluate relative error in computed solution */
        strcpy (fullpathname, testprobs);
        strcat (fullpathname, "Solution/");
        strcat (fullpathname, probname) ;
        /* printf ("reading point: %s\n", fullpathname) ;*/
        f = fopen (fullpathname, "r") ;
        if (f == (FILE *) NULL)
        {
            printf ("file not found\n") ;
            exit (0) ;
        }

        err = 0. ;
        normx = 0. ;
        for (j = 0; j < ncol; j++)
        {
            fscanf (f, "%lg\n", &t) ;
            if ( fabs (t) > normx )
            {
                normx = fabs (t) ;
            }
            if ( fabs (x [j] - t) > err )
            {
                err = fabs (x [j] - t) ;
            }
        }
        fclose (f) ;

        if ( normx > 0. )
        {
            err /= normx ;
        }

        printf ("\n======================================================\n") ;
        printf ("---------- Problem Description ----------\n") ;
        printf ("!!%8s %13.6f %e\n", probname, elapsed, err) ;
        printf ("problem ................................. %s\n", probname) ;
        printf ("   location: %s\n", testprobs) ;
        printf ("solution status ......................... %i\n", status) ;
        printf ("number of rows .......................... %i\n", nrow) ;
        printf ("number of columns ....................... %i\n", ncol) ;
        printf ("number of strict inequalities ........... %i\n", ni) ;
        printf ("run time ................................ %e\n", elapsed) ;
        work = (double *) malloc (2*nrow*sizeof (double)) ;
        t = pproj_KKTerror (&errb, &errx, &absAx, x,lambda, y, nrow, ncol, 1,
                            Ap, Ai, Ax, lo, hi, bl, bu, work) ;
        if ( !Parm.stopabs && (absAx != 0.) )
        {
            errb /= absAx ;
        }
        printf ("\n----------- Error Statistics ------------\n") ;
        printf ("specified tolerance ..................... %e\n", grad_tol) ;
        if ( Parm.stopabs )
        {
            printf ("sup norm of dual function gradient ...... %e\n", errb) ;
        }
        else
        {
            printf ("rel. sup norm of dual function gradient . %e\n", errb) ;
        }
        printf ("relative diff between x & dual minimizer  %e\n", errx) ;
        printf ("absAx ................................... %e\n", absAx) ;

        pproj_print_stat (&Stat, TRUE) ;
        printf ("======================================================\n") ;
        fflush (stdout) ;
#ifndef NDEBUG
        if ( status != 0 )
        {
            pproj_error (-1, __FILE__, __LINE__, "problem not solved\n") ;
        }
#endif
        free (bu) ;
        free (bl) ;
        free (Ap) ;
        free (Ai) ;
        free (Ax) ;
        free (lo) ;
        free (hi) ;
        free (lambda) ;
        free (x) ;
        free (y) ;
        free (work) ;
    }
    fclose (namefile) ;
    return (0) ;
}
