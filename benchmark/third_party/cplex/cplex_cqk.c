// Based on the example qpex1.c from the CPLEX package

#include <ilcplex/cplex.h>
#include <stdlib.h>

int cplex_cqk(int n, double *restrict d, double *restrict a,
    double *restrict b, double r, double *restrict low, double *restrict up,
    double *x, int nthreads, int method, double timelimit)
{
   char     sense[1] = {'E'};
   double   rhs[1] = {r};
   int      beg[1] = {0};
   int      *ind = NULL;

   CPXENVptr     env = NULL;
   CPXLPptr      lp = NULL;
   int           error;
   int           i;

   int           status = -1;

   ind = (int *) malloc(n*sizeof(int));
   if (!ind) goto TERMINATE;

   /* Initialize the CPLEX environment */

   env = CPXopenCPLEX (&error);
   if (env == NULL) goto TERMINATE;

   /* Parameters */

   error = CPXsetintparam (env, CPXPARAM_ScreenOutput, CPX_OFF);
   if (error) goto TERMINATE;
   error = CPXsetintparam (env, CPXPARAM_QPMethod, method);
   if (error) goto TERMINATE;
   error = CPXsetdblparam(env, CPXPARAM_TimeLimit, timelimit);
   if (error) goto TERMINATE;
   error = CPXsetdblparam(env, CPXPARAM_Barrier_ConvergeTol, 1e-8); // relative
   if (error) goto TERMINATE;
   error = CPXsetintparam(env, CPXPARAM_Preprocessing_Presolve, CPX_OFF);
   if (error) goto TERMINATE;
   error = CPXsetintparam(env, CPXPARAM_Threads, nthreads);
   if (error) goto TERMINATE;

   /* Create the problem. */

   lp = CPXcreateprob (env, &error, "cqk");
   if (!lp) goto TERMINATE;

   /* Add vars and the linear part of the objective */
   for (i = 0; i < n; ++i) ind[i] = i;
   error = CPXnewcols(env, lp, n, a, low, up, NULL, NULL);
   if (error) goto TERMINATE;

   /* Linear constraint */
   error = CPXaddrows(env, lp, 0, 1, n, rhs, sense, beg, ind, b, NULL, NULL);
   if (error) goto TERMINATE;

   /* Quadratic terms in the objective */
   for (i = 0; i < n; ++i)
   {
      error = CPXchgqpcoef(env, lp, i, i, d[i]);
      if (error) goto TERMINATE;
   }

   /* Optimize the problem and obtain solution. */

   error = CPXqpopt (env, lp);
   if (error) goto TERMINATE;

   if (CPXgetstat(env, lp) == CPX_STAT_OPTIMAL)
   {
      error = CPXgetx (env, lp, x, 0, n-1);
      if (error) goto TERMINATE;

      if (method == 4)
         status = CPXgetbaritcnt (env, lp);
      else
         status = CPXgetitcnt (env, lp);
   }

TERMINATE:

   /* Free up the problem as allocated by CPXcreateprob, if necessary */

   if (lp != NULL) CPXfreeprob (env, &lp);

   /* Free up the CPLEX environment, if necessary */

   if (env != NULL) CPXcloseCPLEX (&env);

   /* Free up the problem data arrays, if necessary. */

   if (ind) free(ind);

   return status;
}
