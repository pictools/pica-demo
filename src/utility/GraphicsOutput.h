#ifndef PICA_DEMO_UTILITY_GRAPHICS_OUTPUT
#define PICA_DEMO_UTILITY_GRAPHICS_OUTPUT

#include "pica/grid/YeeGrid.h"
#include "pica/math/Dimension.h"
#include "pica/math/Vectors.h"
#include "pica/particles/Ensemble.h"
#include "pica/particles/Particle.h"
#include "pica/particles/ParticleArray.h"
#include "pica/threading/OpenMPHelper.h"
#include "utility/Parameters.h"

#include <ctime>
#include <mgl2/mgl.h>

#if _MSC_VER
#include <direct.h>
#else
#include <sys/stat.h>
#include <sys/types.h>
#endif

using namespace pica;

const std::string currentDateTime()
{
    struct tm tstruct;
    time_t now = time(0);
    tstruct = *localtime(&now);
    char buf[80];
    strftime(buf, sizeof(buf), "%Y-%m-%d_%H-%M-%S", &tstruct);
    return buf;
}

std::string dirDivider()
{
#ifdef _WIN32
    return "\\";
#else
    return "/";
#endif
}

int createDir(const std::string& path)
{
    int ret = 0;
#ifdef _WIN32
    ret = _mkdir(path.c_str());
#else 
    ret = mkdir(path.c_str(), 0777);
#endif
    if (ret == -1)
        std::cerr << "ERROR in mkdir(" << path << "):  " << strerror(errno) << "\n";

    return ret;
}

template<class Ensemble>
void getParticlesDensity(const Ensemble& ensemble, int typeIndex, mglData& density1d, mglData& density2d);

template<class ParticleArray>
void getParticlesDensity(const EnsembleSupercells<ParticleArray>& ensemble, int typeIndex, mglData& density1d, mglData& density2d)
{
    Int3 numSupercells = ensemble.getNumSupercells();
    #pragma omp parallel for collapse(3)
    for (int i = 0; i < numSupercells.x; i++)
    for (int j = 0; j < numSupercells.y; j++)
    for (int k = 0; k < numSupercells.z; k++)
    {
        const ParticleArray& particles = ensemble.getParticles(pica::Int3(i, j, k));
        for (int idx = 0; idx < particles.size(); idx++)
        {
            ParticleArray::ConstParticleRef particle = particles[idx];
            if (particle.getType() != typeIndex)
                continue;

            pica::FP3 position = particle.getPosition();
            pica::FP3 cf = (position - ensemble.getMinPosition()) / (ensemble.getMaxPosition() - ensemble.getMinPosition());
            int indexX = cf.x * density2d.nx;
            int indexY = cf.y * density2d.ny;
            #pragma omp atomic
            density1d.a[indexX] += particle.getFactor();
            #pragma omp atomic
            density2d.a[indexY * density2d.nx + indexX] += particle.getFactor();
        }
    }
}

template<class ParticleArray>
void getParticlesDensity(const EnsembleUnordered<ParticleArray>& ensemble, int typeIndex, mglData& density1d, mglData& density2d)
{
    const ParticleArray& particles = ensemble.getParticles();
    int numParticles = particles.size();
    #pragma omp parallel for
    for (int i = 0; i < numParticles; i++)
    {
        ParticleArray::ConstParticleRef particle = particles[i];
        if (particle.getType() != typeIndex)
            continue;

        pica::FP3 position = particle.getPosition();
        pica::FP3 cf = (position - ensemble.getMinPosition()) / (ensemble.getMaxPosition() - ensemble.getMinPosition());
        int indexX = cf.x * density2d.nx;
        int indexY = cf.y * density2d.ny;
        #pragma omp atomic
        density1d.a[indexX] += particle.getFactor();
        #pragma omp atomic
        density2d.a[indexY * density2d.nx + indexX] += particle.getFactor();
    }
}

template<class Ensemble, class Grid>
void outputData(Ensemble& particles, Grid& fields,
    const utility::DemoParameters& parameters, int iteration)
{
    typedef pica::Particle<pica::Three> Particle;

    mglGraph gr(0, 1920, 1280);
    mglData ex1d(parameters.outputResolutionWidth);
    mglData ex2d(parameters.outputResolutionWidth, parameters.outputResolutionHeight);

    pica::FieldInterpolatorCIC<YeeGrid<Three, double>> interpolator(fields);

    for (int i = 0; i < parameters.outputResolutionWidth; i++)
    {
        FP3 coords = FP3((double)i / parameters.outputResolutionWidth, 0.5, 0.5)
                * (particles.getMaxPosition() - particles.getMinPosition());
        FP3 electricField, magneticField;
        interpolator.get(coords, electricField, magneticField);
        ex1d[i] = electricField.x;
    }

    for (int j = 0; j < parameters.outputResolutionHeight; j++)
    for (int i = 0; i < parameters.outputResolutionWidth; i++)
    {
        FP3 coords = FP3((double)i / parameters.outputResolutionWidth, (double)j / parameters.outputResolutionHeight, 0.5)
            * (particles.getMaxPosition() - particles.getMinPosition());
        FP3 electricField, magneticField;
        interpolator.get(coords, electricField, magneticField);
        ex2d[j * parameters.outputResolutionWidth + i] = electricField.x;
    }

    gr.SubPlot(2, 2, 0);
    gr.SetRange('x', particles.getMinPosition().x, particles.getMaxPosition().x);
    gr.SetRange('y', -1.2 * parameters.Amp, 1.2 * parameters.Amp);
    gr.Plot(ex1d, "G");
    gr.Label('x', "Ex", 0);
    gr.Label('y', "y", 0);
    gr.Adjust();
    gr.Axis("xy");
    gr.Box();
    gr.Grid("xy", "k:");

    gr.SubPlot(2, 2, 2);
    gr.SetRange('x', particles.getMinPosition().x, particles.getMaxPosition().x);
    gr.SetRange('y', particles.getMinPosition().y, particles.getMaxPosition().y);
    gr.SetRange('c', -1.2 * parameters.Amp, 1.2 * parameters.Amp);
    gr.Dens(ex2d, "BbwrR");
    gr.Label('x', "Ex", 0);
    gr.Label('y', "y", 0);
    gr.Adjust();
    gr.Axis("xy");
    gr.Box();
    gr.Grid("xy", "k:");

    const int typeIndex = 0;
    int outputResolutionWidth = parameters.numCells.x;
    int outputResolutionHeight = parameters.numCells.y;
    mglData electronDensity1d(outputResolutionWidth);
    mglData electronDensity2d(outputResolutionWidth, outputResolutionHeight);
    getParticlesDensity(particles, typeIndex, electronDensity1d, electronDensity2d);

    double minValue = 0.9 * parameters.NumPerCell * parameters.numCells.z * parameters.numCells.y * parameters.particlesFactor;
    double maxValue = 1.1 * parameters.NumPerCell * parameters.numCells.z * parameters.numCells.y * parameters.particlesFactor;

    gr.SubPlot(2, 2, 1);
    gr.SetRange('x', particles.getMinPosition().x, particles.getMaxPosition().x);
    gr.SetRange('y', minValue, maxValue);
    gr.Plot(electronDensity1d, "G");
    gr.Label('x', "Electron density", 0);
    gr.Label('y', "y", 0);
    gr.Adjust();
    gr.Axis("xy");
    gr.Box();
    gr.Grid("xy", "k:");

    minValue = 0.9 * parameters.NumPerCell * parameters.numCells.z * parameters.particlesFactor;
    maxValue = 1.1 * parameters.NumPerCell * parameters.numCells.z * parameters.particlesFactor;
    gr.SubPlot(2, 2, 3);
    gr.SetRange('x', particles.getMinPosition().x, particles.getMaxPosition().x);
    gr.SetRange('y', particles.getMinPosition().y, particles.getMaxPosition().y);
    gr.SetRange('c', minValue, maxValue);
    gr.Dens(electronDensity2d, "BbwrR");
    gr.Label('x', "Electron Density", 0);
    gr.Label('y', "y", 0);
    gr.Adjust();
    gr.Axis("xy");
    gr.Box();
    gr.Grid("xy", "k:");
    gr.Colorbar("BbwrR");

    std::ostringstream ss;
    ss << parameters.outputDir << dirDivider() <<  "iteration_" << iteration << ".png";
    gr.WritePNG(ss.str().c_str());
}

#endif
