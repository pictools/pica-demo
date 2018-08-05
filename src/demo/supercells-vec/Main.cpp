#include "utility/BoundaryConditions.h"
#include "utility/FieldGenerator.h"
#include "utility/GraphicsOutput.h"
#include "utility/Parameters.h"
#include "utility/ParticleGenerator.h"
#include "utility/Random.h"
#include "utility/Timer.h"

#include "pica/currentDeposition/CurrentDepositor.h"
#include "pica/fieldInterpolation/FieldInterpolator.h"
#include "pica/fieldSolver/YeeSolver.h"
#include "pica/grid/YeeGrid.h"
#include "pica/math/Dimension.h"
#include "pica/math/Vectors.h"
#include "pica/particles/Ensemble.h"
#include "pica/particles/Particle.h"
#include "pica/particles/ParticleArray.h"
#include "pica/particlePush/BorisPusher.h"
#include "pica/threading/OpenMPHelper.h"

#include <mgl2/mgl.h>

#include <algorithm>
#include <iostream>
#include <memory>

using namespace pica;

namespace pica {
    namespace ParticleTypes {
        std::vector<ParticleType> typesVector;
        const ParticleType* types = NULL;
        int numTypes = 0;
    }
}

template<class Ensemble, class Grid>
void runDemo(Ensemble& particles, Grid& fields, const utility::DemoParameters& parameters);

template<class Ensemble, class Grid>
void runIteration(Ensemble& particles, Grid& fields, Ensemble& migratingParticles, double dt);

template<class Ensemble, class Grid>
void updateParticles(Ensemble& particles, Grid& fields, Ensemble& migratingParticles, double dt);

template<class Grid>
void zeroizeCurrents(Grid& fields);

template<class Ensemble, class Grid>
void process(Ensemble& particles, Grid& fields, Ensemble& migratingParticles,
    pica::Int3 supercellIdx, double dt);

template<class Ensemble>
void migrateAndApplyBoundaryConditions(Ensemble& particles, Ensemble& migratingParticles, pica::Int3 supercellIdx);

template<class Ensemble, class Grid>
void push(Ensemble& particles, const Grid& fields, pica::Int3 supercellIdx, double dt);

template<class Ensemble, class Grid>
void depositCurrents(Ensemble& particles, Grid& fields,
    pica::Int3 supercellIdx, double dt);

template<class Grid>
void updateFields(Grid& fields, double dt);

int main(int argc, char* argv[])
{
    typedef Particle<Three> Particle;
    typedef ParticleArrayAoS<Particle> ParticleArray;
    typedef typename Ensemble<ParticleArray, EnsembleRepresentation_Supercells>::Type Ensemble;
    typedef YeeGrid<Three> Grid;

    utility::DemoParameters parameters = utility::getParameters();
    parameters.outputDir = "demo_" + currentDateTime();

    Ensemble particles(parameters.minPosition, parameters.maxPosition, parameters.numCells, parameters.numCellsPerSupercell);
    utility::generateParticles(parameters, particles);
    Grid fields = utility::generateField<Grid>(parameters);

    std::auto_ptr<utility::Stopwatch> timer(utility::createStopwatch());
    timer->start();
    runDemo(particles, fields, parameters);
    timer->stop();

    std::cout << "Total run time: " << timer->getElapsed() << " sec";

    return 0;
}

// Run the whole benchmark
template<class Ensemble, class Grid>
void runDemo(Ensemble& particles, Grid& fields,
    const utility::DemoParameters& parameters)
{
    omp_set_num_threads(parameters.numThreads);
    Ensemble migratingParticles(particles.getMinPosition(), particles.getMaxPosition(),
        particles.getNumCells(), particles.getNumCellsPerSupercell());

    createDir(parameters.outputDir);
    for (int iteration = 0; iteration < parameters.numIterations; iteration++)
    {
        if (parameters.outputPeriod > 0 && iteration % parameters.outputPeriod == 0)
            outputData(particles, fields, parameters, iteration);
        runIteration(particles, fields, migratingParticles, parameters.dt);
    }
}


// Simulate one time step
template<class Ensemble, class Grid>
void runIteration(Ensemble& particles, Grid& fields, Ensemble& migratingParticles, double dt)
{
    zeroizeCurrents(fields);
    updateParticles(particles, fields, migratingParticles, dt);
    handleCurrentsBC(fields);
    updateFields(fields, dt);
}

template<class Ensemble, class Grid>
void updateParticles(Ensemble& particles, Grid& fields, Ensemble& migratingParticles, double dt)
{
    pica::Int3 numSupercells = particles.getNumSupercells();

    pica::Int3 superCellStep(3, 3, 3);
    for (int d = 0; d < 3; d++)
        if (particles.getNumCellsPerSupercell()[d] == 1)
            superCellStep[d] = 4;
    for (int startI = 0; startI < superCellStep.x; startI++)
    for (int startJ = 0; startJ < superCellStep.y; startJ++)
    for (int startK = 0; startK < superCellStep.z; startK++) {
        #pragma omp parallel for collapse(3) schedule(static, 1)
        for (int i = startI; i < numSupercells.x; i += superCellStep.x)
        for (int j = startJ; j < numSupercells.y; j += superCellStep.y)
        for (int k = startK; k < numSupercells.z; k += superCellStep.z)
            process(particles, fields, migratingParticles, pica::Int3(i, j, k), dt);
    }

    // Finalize migration
    #pragma omp parallel for collapse(3)
    for (int i = 0; i < numSupercells.x; i++)
    for (int j = 0; j < numSupercells.y; j++)
    for (int k = 0; k < numSupercells.z; k++) {
        pica::ParticleArrayAoS<pica::Particle<pica::Three> >& migrating = migratingParticles.getParticles(pica::Int3(i, j, k));
        for (int idx = 0; idx < migrating.size(); idx++)
            particles.add(migrating[idx]);
        int size = migrating.size();
        for (int idx = 0; idx < size; idx++)
            migrating.popBack();
    }
}

template<class Ensemble, class Grid>
void process(Ensemble& particles, Grid& fields, Ensemble& migratingParticles,
    pica::Int3 supercellIdx, double dt)
{
    push(particles, fields, supercellIdx, dt);
    migrateAndApplyBoundaryConditions(particles, migratingParticles, supercellIdx);
    depositCurrents(particles, fields, supercellIdx, dt);
}

template<class Ensemble, class Grid>
void push(Ensemble& particles, const Grid& fields, pica::Int3 supercellIdx, double dt)
{
    typedef typename Ensemble::Particle Particle;
    pica::ParticleArrayAoS<Particle>& particleArray = particles.getParticles(supercellIdx);
    pica::BorisPusher<Particle, double> pusher(dt);
    pica::FP3 supercellMinPosition = particles.getMinPosition() +
        fields.getStep() * pica::FP3(supercellIdx * particles.getNumCellsPerSupercell());
    pica::FieldInterpolatorCICSupercell<double> fieldInterpolator(fields,
        supercellMinPosition, particles.getNumCellsPerSupercell());

    static const int tileSize = 16;
    pica::Vector3<double> e[tileSize];
    pica::Vector3<double> b[tileSize];
    pica::Vector3<double> pos[tileSize];
    const int numParticles = particleArray.size();
    const int numTiles = numParticles / tileSize;
    for (int tileIdx = 0; tileIdx < numTiles; tileIdx++) {
        const int startIdx = tileIdx * tileSize;

        for (int i = 0; i < tileSize; i++)
            pos[i] = particleArray[i + startIdx].getPosition();

#pragma forceinline recursive
#pragma ivdep
#pragma vector always
        for (int i = 0; i < tileSize; i++)
            fieldInterpolator.get(pos[i], e[i], b[i]);

#pragma forceinline
#pragma ivdep
#pragma vector always
        for (int i = 0; i < tileSize; i++)
            pusher.push(&particleArray[i + startIdx], e[i], b[i]);
    }

    const int startIdx = numTiles * tileSize;
    const int endIdx = std::min<int>(startIdx + tileSize, numParticles);

    for (int i = startIdx; i < endIdx; i++)
        pos[i - startIdx] = particleArray[i].getPosition();

#pragma forceinline recursive
#pragma ivdep
#pragma vector always
    for (int i = startIdx; i < endIdx; i++)
        fieldInterpolator.get(pos[i - startIdx], e[i - startIdx], b[i - startIdx]);

#pragma forceinline
#pragma ivdep
#pragma vector always
    for (int i = startIdx; i < endIdx; i++)
        pusher.push(&particleArray[i], e[i - startIdx], b[i - startIdx]);
}


// Reflective boundary conditions
template<class Ensemble>
void migrateAndApplyBoundaryConditions(Ensemble& particles, Ensemble& migratingParticles, pica::Int3 supercellIdx)
{
    typedef typename Ensemble::Particle Particle;
    pica::ParticleArrayAoS<Particle>& particleArray = particles.getParticles(supercellIdx);
    pica::Vector3<double> minPosition = particles.getMinPosition();
    pica::Vector3<double> maxPosition = particles.getMaxPosition();
    for (int i = 0; i < particleArray.size(); i++) {
        pica::Vector3<double> position = particleArray[i].getPosition();
        particleArray[i].setPosition(position);
        pica::Vector3<double> momentum = particleArray[i].getMomentum();
        particleArray[i].setMomentum(momentum);
        // First apply boundary conditions as it could change the position
        // Non-migrating particles are definitely not subject for BC
        if (particles.getSupercellIndex(particleArray[i]) != supercellIdx) {
            pica::Vector3<double> position = particleArray[i].getPosition();
            for (int d = 0; d < 3; d++)
                if (position[d] < minPosition[d]) {
                    position[d] += (maxPosition[d] - minPosition[d]);
                }
                else
                    if (position[d] > maxPosition[d]) {
                        position[d] -= (maxPosition[d] - minPosition[d]);
                    }
            particleArray[i].setPosition(position);
        }
        pica::Int3 newSupercellIdx = particles.getSupercellIndex(particleArray[i]);
        // Now check again that migration happens after applying boundary conditions
        if (newSupercellIdx != supercellIdx) {
            migratingParticles.add(particleArray[i]);
            typename Ensemble::ParticleRef lastParticle = particleArray.back();
            particleArray[i] = lastParticle;
            particleArray.popBack();
            i--;
        }
    }
}

template<class Grid>
void zeroizeCurrents(Grid& fields)
{
    typedef typename Grid::IndexType IndexType;
    IndexType gridSize = fields.getSize();
    #pragma omp parallel for collapse(3)
    for (int i = 0; i < gridSize.x; i++)
    for (int j = 0; j < gridSize.y; j++)
    for (int k = 0; k < gridSize.z; k++) {
        fields.jx(i, j, k) = 0.0;
        fields.jy(i, j, k) = 0.0;
        fields.jz(i, j, k) = 0.0;
    }
}

template<class Ensemble, class Grid>
void depositCurrents(Ensemble& particles, Grid& fields,
    pica::Int3 supercellIdx, double dt)
{
    typedef typename Ensemble::Particle Particle;
    pica::ParticleArrayAoS<Particle>& particleArray = particles.getParticles(supercellIdx);
    const double halfDt = 0.5 * dt;
    pica::FP3 supercellMinPosition = particles.getMinPosition() +
        fields.getStep() * pica::FP3(supercellIdx * particles.getNumCellsPerSupercell());
    pica::CurrentDepositorCICSupercell<double> currentDepositor(fields, supercellMinPosition, particles.getNumCellsPerSupercell());
    const int numParticles = particleArray.size();
    for (int i = 0; i < numParticles; i++) {
        FP3 velocity = particleArray[i].getVelocity();
        FP3 position = particleArray[i].getPosition() - velocity * halfDt;
        pica::Vector3<double> current = velocity * particleArray[i].getFactor() * particleArray[i].getCharge();
        currentDepositor.deposit(position, current);
    }
}

template<class Grid>
void updateFields(Grid& fields, double dt)
{
    pica::YeeSolver solver;
    solver.updateB(fields, dt / 2.0);
    handleMagneticFieldBC(fields);
    solver.updateE(fields, dt);
    handleElectricFieldBC(fields);
    solver.updateB(fields, dt / 2.0);
    handleMagneticFieldBC(fields);
}
