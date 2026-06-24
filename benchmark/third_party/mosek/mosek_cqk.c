// Based on the example qo1.c from the MOSEK package

#include <stdio.h>
#include "mosek.h" /* Include the MOSEK definition file. */

// Uncomment for debugging
// static void MSKAPI printstr(void *handle,
//                             const char *str)
// {
//     printf("%s", str);
//     fflush(stdout);
// }

typedef struct {
  MSKenv_t  env;
  MSKtask_t task;
} MOSEK_model;

void MOSEK_model_free(MOSEK_model *model)
{
  if (model->task) MSK_deletetask(&model->task);
  if (model->env) MSK_deleteenv(&model->env);
  if (model) free(model);
  model = NULL;
}

MOSEK_model *MOSEK_model_create(
   int n,
   double *restrict b,
   double r,
   double *restrict low,
   double *restrict up,
   int nthreads,
   double timelimit
)
{
  MOSEK_model *model = malloc(sizeof(MOSEK_model));

  MSKint32t     j;
  MSKrescodee   error;
  char sym[MSK_MAX_STR_LEN];
  char desc[MSK_MAX_STR_LEN];

  /* Create the mosek environment. */
  error = MSK_makeenv(&model->env, NULL);
  if (error != MSK_RES_OK) goto ERROR;

  /* Create the optimization task. */
  error = MSK_maketask(model->env, 1, n, &model->task);
  if (error != MSK_RES_OK) goto ERROR;

  // Uncomment for debugging
  // MSK_linkfunctotaskstream(model->task, MSK_STREAM_LOG, NULL, printstr);

  // Parameters
  error = MSK_putintparam(model->task, MSK_IPAR_LOG, 0);
  if (error != MSK_RES_OK) goto ERROR;
  error = MSK_putintparam(model->task, MSK_IPAR_OPTIMIZER, MSK_OPTIMIZER_INTPNT);
  if (error != MSK_RES_OK) goto ERROR;
  error = MSK_putintparam(model->task, MSK_IPAR_NUM_THREADS, nthreads);
  if (error != MSK_RES_OK) goto ERROR;
  error = MSK_putintparam(model->task, MSK_IPAR_PRESOLVE_USE, MSK_OFF);
  if (error != MSK_RES_OK) goto ERROR;
  error = MSK_putintparam(model->task, MSK_IPAR_INTPNT_SCALING, MSK_SCALING_NONE);
  if (error != MSK_RES_OK) goto ERROR;
  error = MSK_putdouparam(model->task, MSK_DPAR_OPTIMIZER_MAX_TIME, timelimit);
  if (error != MSK_RES_OK) goto ERROR;
  error = MSK_putintparam(model->task, MSK_IPAR_INTPNT_HOTSTART, MSK_OFF);
  if (error != MSK_RES_OK) goto ERROR;
  error = MSK_putintparam(model->task, MSK_IPAR_INTPNT_BASIS, MSK_BI_NEVER);
  if (error != MSK_RES_OK) goto ERROR;

  /* Append 1 empty constraint.
    The constraint will initially have no bounds. */
  error = MSK_appendcons(model->task, 1);
  if (error != MSK_RES_OK) goto ERROR;

  /* Append n variables.
    The variables will initially be fixed at zero (x=0). */
  error = MSK_appendvars(model->task, n);
  if (error != MSK_RES_OK) goto ERROR;

  // Set constraint terms
  for (j = 0; j < n; ++j)
  {
    error = MSK_putaij(model->task, 0, j, b[j]);
    if (error != MSK_RES_OK) goto ERROR;
  }

  // Set variables bounds
  for (j = 0; j < n; ++j)
  {
    error = MSK_putvarbound(model->task, j, MSK_BK_RA, low[j], up[j]);
    if (error != MSK_RES_OK) goto ERROR;
  }

  // Set the bounds on constraint
  error = MSK_putconbound(model->task, 0, MSK_BK_FX, r, r);
  if (error != MSK_RES_OK) goto ERROR;

  return model;

  ERROR:

  // Print error message
  MSK_getcodedesc(error, sym, desc);
  printf("%s (%d): %s\n", sym, error, desc);

  MOSEK_model_free(model);

  return NULL;
}

int mosek_cqk(
  MOSEK_model *model,
  int n,
  double *restrict d,
  double *restrict a,
  double *x
)
{
  int status = -1;

  MSKint32t     j;
  MSKrescodee   error;
  MSKsolstae    solsta;
  MSKrescodee   trmcode;

  // delete previous solution
  error = MSK_deletesolution(model->task, MSK_SOL_ITR);
  if (error != MSK_RES_OK) goto QUIT;

  for (j = 0; j < n; ++j)
  {
    // linear term in the objective
    error = MSK_putcj(model->task, j, -a[j]);
    if (error != MSK_RES_OK) goto QUIT;

    // quadratic term in the objective
    error = MSK_putqobjij(model->task, j, j, d[j]);
    if (error != MSK_RES_OK) goto QUIT;
  }

  /* Run optimizer */
  error = MSK_optimizetrm(model->task, &trmcode);
  if (error != MSK_RES_OK) goto QUIT;

  error = MSK_getsolsta(model->task, MSK_SOL_ITR, &solsta);
  if (error != MSK_RES_OK) goto QUIT;

  if (solsta == MSK_SOL_STA_OPTIMAL)
  {
      error = MSK_getxx(model->task, MSK_SOL_ITR, x);
      if (error != MSK_RES_OK) goto QUIT;

      MSK_getintinf(model->task, MSK_IINF_INTPNT_ITER, &status);
  }

  QUIT:

  return status;
}
