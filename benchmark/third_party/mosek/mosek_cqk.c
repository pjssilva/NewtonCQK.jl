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

int mosek_cqk(
   int n,
   double *restrict d,
   double *restrict a,
   double *restrict b,
   double rhs,
   double *restrict low,
   double *restrict up,
   int *restrict inds,
   double *x,
   int nthreads,
   double timelimit
)
{
  int status = -1;

  MSKint32t     j;
  MSKenv_t      env = NULL;
  MSKtask_t     task = NULL;
  MSKrescodee   r;
  MSKsolstae    solsta;
  MSKrescodee   trmcode;

  /* Create the mosek environment. */
  r = MSK_makeenv(&env, NULL);
  if (r != MSK_RES_OK) goto TERMINATE;

  /* Create the optimization task. */
  r = MSK_maketask(env, 1, n, &task);
  if (r != MSK_RES_OK) goto TERMINATE;

  // Uncomment for debugging
  // MSK_linkfunctotaskstream(task, MSK_STREAM_LOG, NULL, printstr);

  // r = MSK_putdouparam(task, MSK_DPAR_INTPNT_QO_TOL_INFEAS, 1e-9);
  r = MSK_putintparam(task, MSK_IPAR_LOG, 0);
  if (r != MSK_RES_OK) goto TERMINATE;
  r = MSK_putintparam(task, MSK_IPAR_OPTIMIZER, MSK_OPTIMIZER_INTPNT);
  if (r != MSK_RES_OK) goto TERMINATE;
  r = MSK_putintparam(task, MSK_IPAR_NUM_THREADS, nthreads);
  if (r != MSK_RES_OK) goto TERMINATE;
  r = MSK_putintparam(task, MSK_IPAR_PRESOLVE_USE, MSK_OFF);
  if (r != MSK_RES_OK) goto TERMINATE;
  r = MSK_putintparam(task, MSK_IPAR_INTPNT_SCALING, MSK_SCALING_NONE);
  if (r != MSK_RES_OK) goto TERMINATE;
  r = MSK_putdouparam(task, MSK_DPAR_OPTIMIZER_MAX_TIME, timelimit);
  if (r != MSK_RES_OK) goto TERMINATE;

  /* Append 1 empty constraint.
    The constraint will initially have no bounds. */
  r = MSK_appendcons(task, 1);
  if (r != MSK_RES_OK) goto TERMINATE;

  /* Append n variables.
    The variables will initially be fixed at zero (x=0). */
  r = MSK_appendvars(task, n);
  if (r != MSK_RES_OK) goto TERMINATE;

  // Set constraint terms
  r = MSK_putarow(task, 0, n, inds, b);
  if (r != MSK_RES_OK) goto TERMINATE;

  /* Set the linear term in the objective.*/
  r = MSK_putclist(task, n, inds, a);
  if (r != MSK_RES_OK) goto TERMINATE;

  // quadratic term in the objective function
  r = MSK_putqobj(task, n, inds, inds, d);
  if (r != MSK_RES_OK) goto TERMINATE;

  // variables bounds
  for (j = 0; j < n; ++j)
  {
    r = MSK_putvarbound(task, j, MSK_BK_RA, low[j], up[j]);
    if (r != MSK_RES_OK) goto TERMINATE;
  }

  /* Set the bounds on constraint */
  r = MSK_putconbound(task, 0, MSK_BK_FX, rhs, rhs);
  if (r != MSK_RES_OK) goto TERMINATE;

  /* Run optimizer */
  r = MSK_optimizetrm(task, &trmcode);
  if (r != MSK_RES_OK) goto TERMINATE;

  r = MSK_getsolsta(task, MSK_SOL_ITR, &solsta);
  if (r != MSK_RES_OK) goto TERMINATE;

  if (solsta == MSK_SOL_STA_OPTIMAL)
  {
      r = MSK_getxx(task, MSK_SOL_ITR, x);
      if (r != MSK_RES_OK) goto TERMINATE;

      MSK_getintinf(task, MSK_IINF_INTPNT_ITER, &status);
  }

  TERMINATE:

  if (!task) MSK_deletetask(&task);
  if (!env) MSK_deleteenv(&env);

  return status;
}
