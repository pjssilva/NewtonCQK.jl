// Based on the example toy from the Hexaly package

/*
 * Hexaly warns that using fewer than 4 threads can lead to very poor
 * performance, which we in fact verified. So, we use >= 4 threads.
 * Different runs on the same problem lead to different behaviors. Hexaly is not
 * much customizable; it does not allow us to choose the algorithm, nor does it
 * tell us what is done internally.
 */

#include "optimizer/hexalyoptimizer.h"
#include <iostream>
#include <cstdlib>
#include <vector>

using namespace hexaly;
using namespace std;

extern "C" int hexaly_cqk(int n, double *__restrict d, double *__restrict a,
    double *__restrict b, double r, double *__restrict low, double *__restrict up,
    double *x, int nthreads)
{
    int status = -1;

    try {
        // Declare the optimization model
        HexalyOptimizer optimizer;
        HxModel model = optimizer.getModel();

        // Set parameters
        optimizer.getParam().setSeed(1);
        optimizer.getParam().setTimeLimit(10);
        optimizer.getParam().setVerbosity(0);
        optimizer.getParam().setWarningLevel(0);
        if (nthreads < 4)
            optimizer.getParam().setNbThreads(4);
        else
            optimizer.getParam().setNbThreads(nthreads);

        // variables
        std::vector<HxExpression> y(n);
        for (int i = 0; i < n; ++i)
            y[i] = model.floatVar(low[i], up[i]);

        // linear constraint
        HxExpression knapsack = model.sum();
        for (int i = 0; i < n; ++i)
        {
            HxExpression term = b[i] * y[i];
            knapsack.addOperand(term);
        }

        model.constraint(knapsack == r);

        // objective
        HxExpression obj = model.sum();

        for (int i = 0; i < n; ++i)
        {
            HxExpression qterm = 0.5 * d[i] * y[i] * y[i];
            obj.addOperand(qterm);
        }

        for (int i = 0; i < n; ++i)
        {
            HxExpression lterm = a[i] * y[i];
            obj.addOperand(lterm);
        }

        model.minimize(obj);

        // Close model before solving it
        model.close();

        // Write model (debug only)
        // optimizer.saveEnvironment("cqk.hxm");

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
