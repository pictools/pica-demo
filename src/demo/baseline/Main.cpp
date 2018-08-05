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
void runIteration(Ensemble& particles, Grid& fields, std::vector<Grid>& threadFields, double dt);

template<class Ensemble, class Grid>
void updateParticles(Ensemble& particles, Grid& fields, Ensemble& migratingParticles, double dt);

template<class Grid>
void zeroizeCurrents(Grid& fields);

template<class Ensemble, class Grid>
void process(Ensemble& particles, Grid& fields, Ensemble& migratingParticles,
    pica::Int3 supercellIdx, double dt);

template<class Ensemble>
void applyBoundaryConditions(Ensemble& particles, int beginIdx, int endIdx);

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
    typedef ParticleArraySoA<Particle> ParticleArray;
    typedef typename Ensemble<ParticleArray, EnsembleRepresentation_Unordered>::Type Ensemble;
    typedef YeeGrid<Three> Grid;

    utility::DemoParameters parameters = utility::getParameters();
    parameters.outputDir = "demo_" + currentDateTime();

    Ensemble particles(parameters.minPosition, parameters.maxPosition);
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
    std::vector<Grid> threadFields(parameters.numThreads, fields);

    createDir(parameters.outputDir);
    for (int iteration = 0; iteration < parameters.numIterations; iteration++)
    {
        if (parameters.outputPeriod > 0 && iteration % parameters.outputPeriod == 0)
            outputData(particles, fields, parameters, iteration);
        runIteration(particles, fields, threadFields, parameters.dt);
    }
}


// Simulate one time step
template<class Ensemble, class Grid>
void runIteration(Ensemble& particles, Grid& fields, std::vector<Grid>& threadFields, double dt)
{
    zeroizeCurrents(fields);
    updateParticles(particles, fields, threadFields, dt);
    finalizeCurrents(fields, threadFields);
    handleCurrentsBC(fields);
    updateFields(fields, dt);
}

template<class Ensemble, class Grid>
void updateParticles(Ensemble& particles, const Grid& fields,
    std::vector<Grid>& threadFields, double dt)
{
    // Each thread processes some particles
    const int numParticles = particles.size();
    const int numThreads = pica::getNumThreads();
    const int particlesPerThread = (numParticles + numThreads - 1) / numThreads;
    #pragma omp parallel for
    for (int idx = 0; idx < numThreads; idx++) {
        const int beginIdx = idx * particlesPerThread;
        const int endIdx = std::min(beginIdx + particlesPerThread, numParticles);
        process(particles, fields, threadFields, beginIdx, endIdx, dt);
    }
}

template<class Ensemble, class Grid>
void process(Ensemble& particles, const Grid& fields, std::vector<Grid>& threadFields,
    int beginIdx, int endIdx, double dt)
{
    push(particles, fields, beginIdx, endIdx, dt);
    applyBoundaryConditions(particles, beginIdx, endIdx);
    depositCurrents(particles, threadFields, beginIdx, endIdx, dt);
}

template<class Ensemble, class Grid>
void push(Ensemble& particles, const Grid& fields,
    int beginIdx, int endIdx, double dt)
{
    typedef typename Ensemble::Particle Particle;
    pica::ParticleArraySoA<Particle>& particleArray = particles.getParticles();
    pica::BorisPusher<Particle, double> pusher(dt);
    pica::FieldInterpolatorCIC<Grid> fieldInterpolator(fields);
    //  #pragma omp simd
    #pragma forceinline
    for (int i = beginIdx; i < endIdx; i++) {
        pica::Vector3<double> e, b;
        fieldInterpolator.get(particleArray[i].getPosition(), e, b);
        pusher.push(&particleArray[i], e, b);
    }
}

template<class Ensemble>
void applyBoundaryConditions(Ensemble& particles, int beginIdx, int endIdx)
{
    typedef typename Ensemble::Particle Particle;
    pica::ParticleArraySoA<Particle>& particleArray = particles.getParticles();
    pica::Vector3<double> minPosition = particles.getMinPosition();
    pica::Vector3<double> maxPosition = particles.getMaxPosition();
    for (int i = beginIdx; i < endIdx; i++) {
        pica::Vector3<double> position = particleArray[i].getPosition();
        pica::Vector3<double> momentum = particleArray[i].getMomentum();
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
void depositCurrents(Ensemble& particles, std::vector<Grid>& threadFields,
    int beginIdx, int endIdx, double dt)
{
    Grid& fields = threadFields[omp_get_thread_num()];
    typedef typename Ensemble::Particle Particle;
    pica::ParticleArraySoA<Particle>& particleArray = particles.getParticles();
    const double halfDt = 0.5 * dt;
    pica::CurrentDepositorCIC<Grid> currentDepositor(fields);

    // Zeroise currents
    for (int i = 0; i < fields.getSize().x; i++)
    for (int j = 0; j < fields.getSize().y; j++)
    for (int k = 0; k < fields.getSize().z; k++) {
        fields.jx(i, j, k) = 0.0;
        fields.jy(i, j, k) = 0.0;
        fields.jz(i, j, k) = 0.0;
    }

    for (int i = beginIdx; i < endIdx; i++) {
        pica::Vector3<double> position = particleArray[i].getPosition();
        for (int d = 0; d < 3; d++)
            position[d] -= particleArray[i].getVelocity()[d] * halfDt;
        pica::Vector3<double> current = particleArray[i].getVelocity() *
            particleArray[i].getCharge() * (double)particleArray[i].getFactor();
        currentDepositor.deposit(position, current);
    }
}

template<class Grid>
void finalizeCurrents(Grid& fields, std::vector<Grid>& threadFields)
{
    double normalization = 1.0 / fields.getStep().volume();
    const Int3 size = fields.getSize();

    #pragma omp parallel for collapse(3)
    for (int i = 0; i < size.x; i++)
    for (int j = 0; j < size.y; j++)
    for (int k = 0; k < size.z; k++) {
        fields.jx(i, j, k) = 0.0;
        fields.jy(i, j, k) = 0.0;
        fields.jz(i, j, k) = 0.0;
        for (int threadIdx = 0; threadIdx < threadFields.size(); threadIdx++) {
            const Grid& currentGrid = threadFields[threadIdx];
            fields.jx(i, j, k) += currentGrid.jx(i, j, k);
            fields.jy(i, j, k) += currentGrid.jy(i, j, k);
            fields.jz(i, j, k) += currentGrid.jz(i, j, k);
        }
        fields.jx(i, j, k) *= normalization;
        fields.jy(i, j, k) *= normalization;
        fields.jz(i, j, k) *= normalization;
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
