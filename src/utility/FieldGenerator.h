#ifndef PICA_DEMO_UTILITY_FIELD_GENERATOR_H
#define PICA_DEMO_UTILITY_FIELD_GENERATOR_H


#include "utility/Random.h"
#include "utility/Parameters.h"
#include "pica/math/Vectors.h"

#include <vector>

using pica::Int3;

namespace utility {

template<class Grid>
Grid generateField(const DemoParameters& parameters)
{
    // Generate fields
    int numGhostCells = 2;

    FP3 step = (parameters.maxPosition - parameters.minPosition) / (FP3(parameters.numCells));
    FP3 origin = parameters.minPosition - step * (double)(numGhostCells);
    Int3 numCells = parameters.numCells;
    for (int d = 0; d < 3; d++)
        numCells[d] += 2 * numGhostCells;
    Grid fields(origin, step, numCells);
    Int3 gridSize = fields.getSize();

    for (int i = 0; i < gridSize.x; i++)
    for (int j = 0; j < gridSize.y; j++)
    for (int k = 0; k < gridSize.z; k++)
    {
        fields.ex(i, j, k) = -parameters.Amp * cos(2 * Constants<double>::pi() * (parameters.minPosition.x + (double)i * step.x) / parameters.L);
        fields.ey(i, j, k) = 0;
        fields.ez(i, j, k) = 0;
        fields.bx(i, j, k) = 0;
        fields.by(i, j, k) = 0;
        fields.bz(i, j, k) = 0;
    }

    return fields;
}

} // namespace utility

#endif
