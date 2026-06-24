// Based on the example qc_c from the Gurobi package

#include <stdlib.h>
#include <stdio.h>
#include "gurobi_c.h"

typedef struct {
  GRBenv    *env;
  GRBmodel  *model;
  int       *inds;
} GUROBI_model;

void GUROBI_model_free(GUROBI_model *model)
{
  GRBfreemodel(model->model);
  GRBfreeenv(model->env);
  if (model->inds) free(model->inds);
  if (model) free(model);
  model = NULL;
}

GUROBI_model *GUROBI_model_create(
  int n,
  double *restrict b,
  double r,
  double *restrict low,
  double *restrict up,
  int nthreads,
  double timelimit
)
{
  GUROBI_model *model = malloc(sizeof(GUROBI_model));

  int i;
  int error = 0;

  /* Create environment */
  error = GRBemptyenv(&model->env);
  if (error) goto ERROR;

  /* Parameters */
  error = GRBsetintparam(model->env, "Seed", 0);
  if (error) goto ERROR;
  error = GRBsetintparam(model->env, "Crossover", 0);
  if (error) goto ERROR;
  error = GRBsetintparam(model->env, "Method", 2);
  if (error) goto ERROR;
  error = GRBsetdblparam(model->env, "TimeLimit", timelimit);
  if (error) goto ERROR;
  error = GRBsetintparam(model->env, "ScaleFlag", 0);
  if (error) goto ERROR;
  error = GRBsetintparam(model->env, "Presolve", 0);
  if (error) goto ERROR;
  error = GRBsetintparam(model->env, "Threads", nthreads);
  if (error) goto ERROR;
  error = GRBsetintparam(model->env, "OutputFlag", 0);
  if (error) goto ERROR;
  // GRBsetdblparam(GRBstartenv, "BarConvTol", 1e-8);     // Barrier convergence tolerance (def 1e-8)
  // GRBsetdblparam(GRBstartenv, "FeasibilityTol" 1e-6);  // Primal feasibility tolerance (def 1e-6)

  /* Start environment */
  error = GRBstartenv(model->env);
  if (error) goto ERROR;

  /* Create the model */

  model->inds = malloc(n * sizeof(int));
  if (!model->inds) goto ERROR;

  for (i = 0; i < n; ++i) model->inds[i] = i;

  error = GRBnewmodel(
    model->env, &model->model, "cqk", n, NULL, low, up, NULL, NULL
  );
  if (error) goto ERROR;
  /* Linear constraint */

  error = GRBaddconstr(model->model, n, model->inds, b, GRB_EQUAL, r, NULL);
  if (error) goto ERROR;

  error = GRBupdatemodel(model->model);
  if (error) goto ERROR;

  return model;

  ERROR:

  // Print error message
  printf("%s\n", GRBgeterrormsg(model->env));

  GUROBI_model_free(model);

  return NULL;
}

int gurobi_cqk(
  GUROBI_model *model,
  int n,
  double *restrict d,
  double *restrict a,
  double *x
)
{
  int       i;
  int       error = 0;
  int       optimstatus;
  int       status = -1;

  /* Reset previous solution information */
  error = GRBreset(model->model, 1);
  if (error) goto QUIT;

  /* Quadratic objective terms */
  error = GRBaddqpterms(model->model, n, model->inds, model->inds, d);
  if (error) goto QUIT;

  /* Linear objective terms */
  for (i = 0; i < n; ++i)
  {
    error = GRBsetdblattrelement(model->model, "Obj", i, -a[i]);
    if (error) goto QUIT;
  }

  /* Update model */
  error = GRBupdatemodel(model->model);
  if (error) goto QUIT;

  /* Optimize model */
  error = GRBoptimize(model->model);
  if (error) goto QUIT;

  /* Capture solution information */
  error = GRBgetintattr(model->model, GRB_INT_ATTR_STATUS, &optimstatus);
  if (error) goto QUIT;

  if (optimstatus == GRB_OPTIMAL)
  {
    error = GRBgetdblattrarray(model->model, GRB_DBL_ATTR_X, 0, n, x);
    if (error) goto QUIT;

    error = GRBgetintattr(model->model, GRB_INT_ATTR_BARITERCOUNT, &status);
    if (error) goto QUIT;
  }

  QUIT:

  return status;
}
