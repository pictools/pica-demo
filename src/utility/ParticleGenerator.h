#ifndef PICA_DEMO_UTILITY_PARTICLE_GENERATOR_H
#define PICA_DEMO_UTILITY_PARTICLE_GENERATOR_H


#include "pica/math/Constants.h"
#include "pica/math/Vectors.h"
#include "pica/particles/Particle.h"
#include "utility/Parameters.h"

#include <cmath>

namespace utility {

template<class Ensemble>
void generateParticles(const DemoParameters& parameters, Ensemble& particles)
{
    ParticleType electron;
    electron.mass = constants::electronMass;
    electron.charge = constants::electronCharge;
    ParticleTypes::numTypes = 1;
    ParticleTypes::typesVector.resize(ParticleTypes::numTypes);
    ParticleTypes::typesVector[0] = electron;
    ParticleTypes::types = &ParticleTypes::typesVector[0];
    FP3 step = (parameters.maxPosition - parameters.minPosition) / (FP3(parameters.numCells));
    utility::Random random;
    for (int i = 0; i < parameters.numCells.x; i++)
    for (int j = 0; j < parameters.numCells.y; j++)
    for (int k = 0; k < parameters.numCells.z; k++)
    {
        double expectedNumParticles = parameters.Density * step.volume() / parameters.particlesFactor
            * (1 + parameters.A * sin(2 * Constants<double>::pi() * (parameters.minPosition.x + ((double)i + 0.5) * step.x) / parameters.L));
        int numParticles = int(expectedNumParticles);
        if (random.getUniform() < expectedNumParticles - numParticles)
            ++numParticles;

        for (int idx = 0; idx < numParticles; idx++)
        {
            typename Ensemble::Particle particle;
            particle.setType(rand() % ParticleTypes::numTypes);
            FP3 position;
            FP3 cellMinPosition = parameters.minPosition + step * FP3(i, j, k);
            FP3 cellMaxPosition = cellMinPosition + step;
            for (int d = 0; d < 3; d++)
                position[d] = cellMinPosition[d] + step[d] * random.getUniform();
            particle.setPosition(position);
            // The standard deviation is sqrt(1/(2*alpha)), where alpha is
            // 3/2 * ((T/mc^2 + 1)^2 - 1)^(-1)
            double temperature = 1e-2 * particle.getMass() * pica::constants::c * pica::constants::c;
            double alpha = temperature / particle.getMass() / pica::constants::c / pica::constants::c + 1;
            alpha = 1.5 / (alpha * alpha - 1);
            double sigma = sqrt(0.5 / alpha) * particle.getMass() * pica::constants::c;
            // Initial particle momentum is combination of given initial
            // momentum based on coords and random term in N(0, sigma)
            FP3 momentum;
            for (int d = 0; d < 3; d++)
                momentum[d] = random.getNormal() * sigma;
            particle.setMomentum(momentum);
            particle.setFactor(parameters.particlesFactor);
            particles.add(particle);
        }
    }
}

} // namespace utility

#endif
