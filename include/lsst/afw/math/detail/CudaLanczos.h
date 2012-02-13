// -*- LSST-C++ -*-

/*
 * LSST Data Management System
 * Copyright 2008 - 2012 LSST Corporation.
 *
 * This product includes software developed by the
 * LSST Project (http://www.lsst.org/).
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the LSST License Statement and
 * the GNU General Public License along with this program.  If not,
 * see <http://www.lsstcorp.org/LegalNotices/>.
 */

/**
 * @file
 *
 * @brief Declaration of a GPU kernel for image warping
 *        and declarations of requred datatypes
 *
 * @note This types have to be specifically declared for GPUs
 *       (since GPUs can't just use of-the shelf libraries)
 *
 * @author Kresimir Cosic
 *
 * @ingroup afw
 */

#ifdef NVCC_COMPILING
    #define CPU_GPU __device__ __host__
#else
    #define CPU_GPU
#endif

namespace lsst {
namespace afw {
namespace math {
namespace detail {
namespace gpu {

typedef lsst::afw::image::VariancePixel VarPixel;
typedef lsst::afw::image::MaskPixel     MskPixel;

const int cWarpingBlockSizeX=16;
const int cWarpingBlockSizeY=16;
const int cWarpingKernelMaxSize=100;

/// Simple 2D point
struct SPoint2
{
    double x;
    double y;

    CPU_GPU SPoint2(double par_x, double par_y) : x(par_x), y(par_y) {}

    #ifndef NVCC_COMPILING
    SPoint2(lsst::afw::geom::Point2D p) : x(p.getX()), y(p.getY()) {}
    #endif
};

/// simple 2D vector
struct SVec2
{
    double x;
    double y;

    CPU_GPU SVec2(double par_x, double par_y) : x(par_x), y(par_y) {}
    CPU_GPU SVec2(SPoint2 a, SPoint2 b) : x(b.x-a.x), y(b.y-a.y) {}

    #ifndef NVCC_COMPILING
    SVec2(lsst::afw::geom::Extent2D e) : x(e.getX()), y(e.getY()) {}
    #endif
};

CPU_GPU inline SVec2 VecAdd(SVec2 a, SVec2 b)
{
    return SVec2(a.x+b.x, a.y+b.y);
}
CPU_GPU inline SVec2 VecSub(SVec2 a, SVec2 b)
{
    return SVec2(a.x-b.x, a.y-b.y);
}
CPU_GPU inline SVec2 VecMul(SVec2 v, double m)
{
    return SVec2(m*v.x, m*v.y);
}
CPU_GPU inline SPoint2 MovePoint(SPoint2 p, SVec2 v)
{
    return SPoint2(p.x+v.x, p.y+v.y);
}


/// defines a 2D range of integer values begX <= x < endX, begY <= y < endY
struct SBox2I
{
    int begX;
    int begY;
    int endX;
    int endY;

    SBox2I() {};

    CPU_GPU SBox2I(int par_begX, int par_begY, int par_endX, int par_endY)
        : begX(par_begX), begY(par_begY), endX(par_endX), endY(par_endY) {}

    CPU_GPU bool isInsideBox(gpu::SPoint2 p)
    {
        return      begX <= p.x && p.x < endX
                 && begY <= p.y && p.y < endY;
    }
};

/** Used for linear interpolation of a 2D function R -> R*R

    This class just defines a line which can be used to interpolate
    a part of a function.

    It does not specify which part of a function is interpolated.
*/
struct LinearInterp
{
    SPoint2 o;    /// defines the value at origin
    SVec2 deltaX; /// difference of neighbouring values in the first column

    LinearInterp(SPoint2 par_o, SVec2 par_deltaX) : o(par_o), deltaX(par_deltaX) {};

    /// Calculates a value of the interpolation function at a point subX
    CPU_GPU SPoint2 Interpolate(int subX)
    {
        return MovePoint(o, VecMul(deltaX,subX) );
    }
};


/** Used for (bi)linear interpolation of a 2D function R*R -> R*R

    This class just defines a 2D plane which can be used to interpolate
    a part of a 2D function.

    It does not specify which part of a function is interpolated.
*/
struct BilinearInterp
{
    SPoint2 o;  /// defines the value at origin
    SVec2 d0X;  /// difference of neighbouring values in the first row
    SVec2 ddX;  /// difference of difference of neighbouring values in two neighbouring rows
    SVec2 deltaY; /// difference of neighbouring values in the first column

    BilinearInterp() : o(0,0), d0X(0,0), ddX(0,0), deltaY(0,0) {};

    CPU_GPU BilinearInterp(SPoint2 par_o, SVec2 par_d0X, SVec2 par_ddX, SVec2 par_deltaY)
        : o(par_o), d0X(par_d0X), ddX(par_ddX), deltaY(par_deltaY) {}


    CPU_GPU LinearInterp GetLinearInterp(int subY)
    {
        SVec2 deltaX=VecAdd(d0X, VecMul(ddX, subY));
        SPoint2 lineBeg= MovePoint(o, VecMul(deltaY,subY) );
        return LinearInterp(lineBeg, deltaX);
    }

    /// Calculates a value of the interpolation plane at a point (subX,subY)
    CPU_GPU SPoint2 Interpolate(int subX, int subY)
    {
        /*SVec2 deltaX = VecAdd(d0X, VecMul(ddX, subY));
        SVec2 diffX  = VecMul(deltaX,subX);
        SVec2 diffY  = VecMul(deltaY,subY);

        //partially mimics original interpolation alg. for warping
        return MovePoint(MovePoint(o, diffY), diffX);*/
        LinearInterp lineY=GetLinearInterp(subY);
        return lineY.Interpolate(subX);
    }
};

/// defines a pixel having image, variance and mask planes
template<typename T>
struct PixelIVM
{
    T img;
    VarPixel var;
    MskPixel msk;
};

/// defines memory region containing image data
template<typename T>
struct ImageDataPtr
{
    T* img;
    VarPixel* var;
    MskPixel* msk;
    int strideImg;
    int strideVar;
    int strideMsk;
    int width;
    int height;
};


template<typename DestPixelT, typename SrcPixelT>
void WarpImageGpuCallKernel(bool isMaskedImage,
                            ImageDataPtr<DestPixelT> destImageGpu,
                            ImageDataPtr<SrcPixelT>  srcImageGpu,
                            int order,
                            SBox2I srcGoodBox,
                            int kernelCenterX,
                            int kernelCenterY,
                            PixelIVM<DestPixelT> edgePixel,
                            SBox2I* srcBlk,
                            BilinearInterp* srcPosInterp,
                            int interpLength
                            );

}}}}} //namespace lsst::afw::math::detail::gpu ends