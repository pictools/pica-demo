#include "pica/math/Vectors.h"

using pica::Int3;

template<class Grid>
void handleElectricFieldBC(Grid& fields)
{
    Int3 gridSize;
    for (int d = 0; d < 3; d++)
    {
        Int3 srcBeginIdx(0, 0, 0);
        Int3 dstBeginIdx(0, 0, 0);
        Int3 size(fields.getSize());

        srcBeginIdx[d] = 2;
        dstBeginIdx[d] = size[d] - 2;
        size[d] = 2;

        for (int i = 0; i < size.x; i++)
        for (int j = 0; j < size.y; j++)
        for (int k = 0; k < size.z; k++)
        {
            Int3 srcIdx = srcBeginIdx + Int3(i, j, k);
            Int3 dstIdx = dstBeginIdx + Int3(i, j, k);
            fields.ex(dstIdx.x, dstIdx.y, dstIdx.z) = fields.ex(srcIdx.x, srcIdx.y, srcIdx.z);
            fields.ey(dstIdx.x, dstIdx.y, dstIdx.z) = fields.ey(srcIdx.x, srcIdx.y, srcIdx.z);
            fields.ez(dstIdx.x, dstIdx.y, dstIdx.z) = fields.ez(srcIdx.x, srcIdx.y, srcIdx.z);
        }

        srcBeginIdx = Int3(0, 0, 0);
        dstBeginIdx = Int3(0, 0, 0);
        size = fields.getSize();

        srcBeginIdx[d] = size[d] - 4;
        dstBeginIdx[d] = 0;
        size[d] = 2;

        for (int i = 0; i < size.x; i++)
        for (int j = 0; j < size.y; j++)
        for (int k = 0; k < size.z; k++)
        {
            Int3 srcIdx = srcBeginIdx + Int3(i, j, k);
            Int3 dstIdx = dstBeginIdx + Int3(i, j, k);
            fields.ex(dstIdx.x, dstIdx.y, dstIdx.z) = fields.ex(srcIdx.x, srcIdx.y, srcIdx.z);
            fields.ey(dstIdx.x, dstIdx.y, dstIdx.z) = fields.ey(srcIdx.x, srcIdx.y, srcIdx.z);
            fields.ez(dstIdx.x, dstIdx.y, dstIdx.z) = fields.ez(srcIdx.x, srcIdx.y, srcIdx.z);
        }
    }
}

template<class Grid>
void handleMagneticFieldBC(Grid& fields)
{
    for (int d = 0; d < 3; d++)
    {
        Int3 srcBeginIdx(0, 0, 0);
        Int3 dstBeginIdx(0, 0, 0);
        Int3 size(fields.getSize());

        srcBeginIdx[d] = 2;
        dstBeginIdx[d] = size[d] - 2;
        size[d] = 2;

        for (int i = 0; i < size.x; i++)
        for (int j = 0; j < size.y; j++)
        for (int k = 0; k < size.z; k++)
        {
            Int3 srcIdx = srcBeginIdx + Int3(i, j, k);
            Int3 dstIdx = dstBeginIdx + Int3(i, j, k);
            fields.bx(dstIdx.x, dstIdx.y, dstIdx.z) = fields.bx(srcIdx.x, srcIdx.y, srcIdx.z);
            fields.by(dstIdx.x, dstIdx.y, dstIdx.z) = fields.by(srcIdx.x, srcIdx.y, srcIdx.z);
            fields.bz(dstIdx.x, dstIdx.y, dstIdx.z) = fields.bz(srcIdx.x, srcIdx.y, srcIdx.z);
        }

        srcBeginIdx = Int3(0, 0, 0);
        dstBeginIdx = Int3(0, 0, 0);
        size = fields.getSize();

        srcBeginIdx[d] = size[d] - 4;
        dstBeginIdx[d] = 0;
        size[d] = 2;

        for (int i = 0; i < size.x; i++)
        for (int j = 0; j < size.y; j++)
        for (int k = 0; k < size.z; k++)
        {
            Int3 srcIdx = srcBeginIdx + Int3(i, j, k);
            Int3 dstIdx = dstBeginIdx + Int3(i, j, k);
            fields.bx(dstIdx.x, dstIdx.y, dstIdx.z) = fields.bx(srcIdx.x, srcIdx.y, srcIdx.z);
            fields.by(dstIdx.x, dstIdx.y, dstIdx.z) = fields.by(srcIdx.x, srcIdx.y, srcIdx.z);
            fields.bz(dstIdx.x, dstIdx.y, dstIdx.z) = fields.bz(srcIdx.x, srcIdx.y, srcIdx.z);
        }
    }
}

template<class Grid>
void handleCurrentsBC(Grid& fields)
{
    for (int d = 0; d < 3; d++)
    {
        Int3 srcBeginIdx(0, 0, 0);
        Int3 dstBeginIdx(0, 0, 0);
        Int3 size(fields.getSize());

        srcBeginIdx[d] = 1;
        dstBeginIdx[d] = size[d] - 3;
        size[d] = 3;

        for (int i = 0; i < size.x; i++)
        for (int j = 0; j < size.y; j++)
        for (int k = 0; k < size.z; k++)
        {
            Int3 srcIdx = srcBeginIdx + Int3(i, j, k);
            Int3 dstIdx = dstBeginIdx + Int3(i, j, k);
            fields.jx(dstIdx.x, dstIdx.y, dstIdx.z) += fields.jx(srcIdx.x, srcIdx.y, srcIdx.z);
            fields.jy(dstIdx.x, dstIdx.y, dstIdx.z) += fields.jy(srcIdx.x, srcIdx.y, srcIdx.z);
            fields.jz(dstIdx.x, dstIdx.y, dstIdx.z) += fields.jz(srcIdx.x, srcIdx.y, srcIdx.z);
            fields.jx(srcIdx.x, srcIdx.y, srcIdx.z) = fields.jx(dstIdx.x, dstIdx.y, dstIdx.z);
            fields.jy(srcIdx.x, srcIdx.y, srcIdx.z) = fields.jy(dstIdx.x, dstIdx.y, dstIdx.z);
            fields.jz(srcIdx.x, srcIdx.y, srcIdx.z) = fields.jz(dstIdx.x, dstIdx.y, dstIdx.z);
        }
    }
}
