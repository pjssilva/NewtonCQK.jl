// Based on the example qpex1.c from the CPLEX package

#include <ilcplex/cplex.h>
#include <stdlib.h>

typedef struct {
   CPXENVptr env;
   CPXLPptr  lp;
} CPLEX_model;

void CPLEX_model_free(CPLEX_model *model)
{
   if (model->lp) CPXfreeprob (model->env, &model->lp);
   if (model->env) CPXcloseCPLEX (&model->env);
   if (model) free(model);
   model = NULL;
}

CPLEX_model *CPLEX_model_create(
   int n,
   double *restrict b,
   double r,
   double *restrict low,
   double *restrict up,
   int nthreads,
   double timelimit
)
{
   CPLEX_model *model = malloc(sizeof(CPLEX_model));

   int      i;
   char     sense[1] = {'E'};
   double   rhs[1] = {r};
   int      beg[1] = {0};
   int      error, status = -1;

   /* Initialize the CPLEX environment */

   model->env = CPXopenCPLEX (&error);
   if (model->env == NULL) goto ERROR;

   /* Parameters */

   error = CPXsetintparam (model->env, CPXPARAM_ScreenOutput, CPX_OFF);
   if (error) goto ERROR;
   error = CPXsetintparam (model->env, CPXPARAM_QPMethod, 4);
   if (error) goto ERROR;
   error = CPXsetdblparam(model->env, CPXPARAM_TimeLimit, timelimit);
   if (error) goto ERROR;
   error = CPXsetdblparam(model->env, CPXPARAM_Barrier_ConvergeTol, 1e-8); // relative
   if (error) goto ERROR;
   error = CPXsetintparam(model->env, CPXPARAM_Preprocessing_Presolve, CPX_OFF);
   if (error) goto ERROR;
   error = CPXsetintparam(model->env, CPXPARAM_Threads, nthreads);
   if (error) goto ERROR;
   error = CPXsetintparam(model->env, CPXPARAM_Advance, 0);
   if (error) goto ERROR;

   /* Create the problem. */

   model->lp = CPXcreateprob (model->env, &error, "cqk");
   if (!model->lp) goto ERROR;

   /* Add vars */
   error = CPXnewcols(model->env, model->lp, n, NULL, low, up, NULL, NULL);
   if (error) goto ERROR;

   /* Linear constraint */
   error = CPXnewrows(model->env, model->lp, 1, rhs, sense, NULL, NULL);
   if (error) goto ERROR;

   for (i = 0; i < n; ++i)
   {
      error = CPXchgcoef(model->env, model->lp, 0, i, b[i]);
      if (error) goto ERROR;
   }

   return model;

   ERROR:

   CPLEX_model_free(model);

   return NULL;
}

int cplex_cqk(
   CPLEX_model *model,
   int n,
   double *restrict d,
   double *restrict a,
   double *x
)
{
   int i;
   int error;
   int status = -1;

   for (i = 0; i < n; ++i)
   {
      /* Linear terms in the objective */
      error = CPXchgcoef(model->env, model->lp, -1, i, -a[i]);
      if (error) goto QUIT;

      /* Quadratic terms in the objective */
      error = CPXchgqpcoef(model->env, model->lp, i, i, d[i]);
      if (error) goto QUIT;
   }

   /* Optimize the problem and obtain solution. */

   error = CPXqpopt (model->env, model->lp);
   if (error) goto QUIT;

   if (CPXgetstat(model->env, model->lp) == CPX_STAT_OPTIMAL)
   {
      error = CPXgetx (model->env, model->lp, x, 0, n-1);
      if (error) goto QUIT;

      status = CPXgetbaritcnt (model->env, model->lp);
   }

   QUIT:

   return status;
}
