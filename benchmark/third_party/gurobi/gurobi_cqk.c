// Based on the example qc_c from the Gurobi package

#include <stdlib.h>
#include <stdio.h>
#include "gurobi_c.h"

int gurobi_cqk(int n, double *restrict d, double *restrict a,
    double *restrict b, double r, double *restrict low, double *restrict up,
    double *x, int nthreads, int method, double timelimit)
{
  GRBenv   *env   = NULL;
  GRBmodel *model = NULL;
  int       error = 0;
  int       *ind;
  int       optimstatus;
  int       status = -1;
  int       i;

  ind = (int *) malloc(n*sizeof(int));
  if (!ind) goto QUIT;

  // The same vector will be used to index vars in the obj and in the const
  for (i = 0; i < n; ++i) ind[i] = i;

  /* Create environment */

  // error = GRBloadenv(&env, NULL);
  error = GRBemptyenv(&env);
  if (error) goto QUIT;

  /* Parameters */

  error = GRBsetintparam(env, "Method", method);
  if (error) goto QUIT;
  error = GRBsetdblparam(env, "TimeLimit", timelimit);
  if (error) goto QUIT;
  error = GRBsetintparam(env, "ScaleFlag", 0);
  if (error) goto QUIT;
  error = GRBsetintparam(env, "Presolve", 0);
  if (error) goto QUIT;
  error = GRBsetintparam(env, "Threads", nthreads);
  if (error) goto QUIT;
  error = GRBsetintparam(env, "OutputFlag", 0);
  if (error) goto QUIT;
  // GRBsetdblparam(env, "BarConvTol", 1e-8);     // Barrier convergence tolerance (def 1e-8)
  // GRBsetdblparam(env, "FeasibilityTol" 1e-6);  // Primal feasibility tolerance (def 1e-6)

  /* Start environment */
  error = GRBstartenv(env);
  if (error) goto QUIT;

  /* Create the model */

  error = GRBnewmodel(env, &model, "cqk", n, a, low, up, NULL, NULL);
  if (error) goto QUIT;

  /* Quadratic objective terms */

  error = GRBaddqpterms(model, n, ind, ind, d);
  if (error) goto QUIT;

  /* Linear constraint */

  error = GRBaddconstr(model, n, ind, b, GRB_EQUAL, r, NULL);
  if (error) goto QUIT;

  /* Optimize model */

  error = GRBoptimize(model);
  if (error) goto QUIT;

  /* Capture solution information */

  error = GRBgetintattr(model, GRB_INT_ATTR_STATUS, &optimstatus);
  if (error) goto QUIT;

  if (optimstatus == GRB_OPTIMAL)
  {
    error = GRBgetdblattrarray(model, GRB_DBL_ATTR_X, 0, n, x);
    if (error) goto QUIT;

    error = GRBgetintattr(model, GRB_INT_ATTR_BARITERCOUNT, &status);
    if (error) goto QUIT;
  }

  QUIT:

  /* Free model */
  GRBfreemodel(model);

  /* Free environment */
  GRBfreeenv(env);

  if (ind) free(ind);

  return status;
}
