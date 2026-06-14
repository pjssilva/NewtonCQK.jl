// Based on the example toy from the Hexaly package

#include "optimizer/hexalyoptimizer.h"
#include <iostream>

using namespace hexaly;
using namespace std;

extern "C" int hexaly_cqk(int n, double *__restrict d, double *__restrict a,
    double *__restrict b, double r, double *__restrict low, double *__restrict up,
    double *x)
{
    int status = -1;

    try {
        // Declare the optimization model
        HexalyOptimizer optimizer;
        HxModel model = optimizer.getModel();

        // variables
        HxExpression* y = (HxExpression*) malloc(n * sizeof(HxExpression));
        for (int i = 0; i < n; ++i)
            y[i] = model.floatVar(low[i], up[i]);

        // linear constraint
        HxExpression knapsack = model.sum();
        for (int i = 0; i < n; ++i)
            knapsack.addOperand(b[i] * y[i]);

        model.constraint(knapsack == r);

        // objective
        HxExpression obj = model.sum();

        for (int i = 0; i < n; ++i)
            obj.addOperand(0.5 * d[i] * y[i] * y[i]);

        for (int i = 0; i < n; ++i)
            obj.addOperand(a[i] * y[i]);

        model.minimize(obj);

        // Close model before solving it
        model.close();

        // Set parameters
        optimizer.getParam().setVerbosity(0);
        optimizer.getParam().setWarningLevel(0);
        optimizer.getParam().setNbThreads(1);

        // Solve
        optimizer.solve();

        if (optimizer.getSolution().getStatus() == SS_Optimal)
        {
            // Return solution
            for (int i = 0; i < n; ++i)
                x[i] = optimizer.getSolution().getDoubleValue(y[i]);

            status = optimizer.getStatistics().getNbIterations();
        }

    } catch (const exception& e) {}

    return status;
}
