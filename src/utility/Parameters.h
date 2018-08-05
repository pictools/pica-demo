#ifndef PICA_DEMO_UTILITY_PARAMETERS_H
#define PICA_DEMO_UTILITY_PARAMETERS_H

#include "pica/math/Vectors.h"
#include "pica/particles/Ensemble.h"
#include "pica/particles/ParticleArray.h"

#include <string>

using pica::Constants;
using pica::FP3;
using pica::Int3;

namespace utility {

struct DemoParameters
{
    Int3 numCells;
    int numIterations;
    int particlesPerCell;
    int numParticleTypes;
    Int3 numCellsPerSupercell;
    int tileSize;
    int numThreads;

    std::string outputDir;

    int outputPeriod;
    int outputResolutionWidth;
    int outputResolutionHeight;

    FP3 minPosition;
    FP3 maxPosition;
    double dt;
    double A;
    double L;
    double NumPerL_Debay;
    int NumPerPlasmaPeriod;
    double NumPerCell;
    int MatrixSize;
    int NumPeriods;
    double SpaceStep;
    double L_Debay;
    double Temp;
    double Density;
    double w_p;
    double Amp;
    double particlesFactor;
};

DemoParameters getParameters()
{
    DemoParameters parameters;
    parameters.A = 0.05;
    parameters.L = 1.0;
    parameters.NumPerL_Debay = 0.5;
    parameters.NumPerPlasmaPeriod = 256;
    parameters.NumPerCell = 30;
    parameters.MatrixSize = 64;
    parameters.NumPeriods = parameters.MatrixSize / (2.0 * sqrt(2.0) * Constants<double>::pi() * parameters.NumPerL_Debay);
    parameters.SpaceStep = parameters.L / parameters.MatrixSize;
    parameters.L_Debay = parameters.SpaceStep * parameters.NumPerL_Debay;
    parameters.Temp = 1e-2 * Constants<double>::electronMass() * Constants<double>::c() * Constants<double>::c();
    parameters.Density = parameters.Temp / (8 * Constants<double>::pi() * Constants<double>::electronCharge() * parameters.L_Debay * Constants<double>::electronCharge() * parameters.L_Debay);
    parameters.w_p = sqrt(4 * Constants<double>::pi() * Constants<double>::electronCharge() * Constants<double>::electronCharge() * parameters.Density / Constants<double>::electronMass());
    parameters.dt = 2 * (Constants<double>::pi() / parameters.w_p) / parameters.NumPerPlasmaPeriod;
    parameters.Amp = 2 * parameters.L * parameters.Density * Constants<double>::electronCharge() * parameters.A;

    parameters.numCells = Int3(parameters.MatrixSize, parameters.MatrixSize / 8, parameters.MatrixSize / 8);
    parameters.numIterations = parameters.NumPerPlasmaPeriod * parameters.NumPeriods;
    parameters.outputPeriod = 16;
    parameters.outputResolutionWidth = 256;
    parameters.outputResolutionHeight = 256;
    parameters.numCellsPerSupercell = Int3(2, 2, 2);
    parameters.minPosition = FP3(0.0, 0.0, 0.0);
    parameters.maxPosition = FP3(parameters.L, parameters.L / 8.0, parameters.L / 8.0);
    FP3 step = (parameters.maxPosition - parameters.minPosition) / (FP3(parameters.numCells));
    parameters.particlesFactor = parameters.Density * step.volume() / parameters.NumPerCell;
    parameters.numThreads = omp_get_max_threads();
    return parameters;
}

} // namespace utility


#endif
