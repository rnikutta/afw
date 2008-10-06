// -*- LSST-C++ -*-
/**
 * @file
 *
 * @brief Definition of functions declared in ConvolveImage.h
 *
 * This file is meant to be included by lsst/afw/math/KernelFunctions.h
 *
 * @todo
 * * Speed up convolution
 *
 * @note: the convolution and apply functions assume that data in a row is contiguous,
 * both in the input image and in the kernel. This will eventually be enforced by afw.
 *
 * @author Russell Owen
 *
 * @ingroup afw
 */
#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <vector>
#include <string>

#include "boost/format.hpp"

#include "lsst/pex/logging/Trace.h"
#include "lsst/afw/image/ImageUtils.h"
#include "lsst/afw/image/MaskedImage.h"

// if true, ignore kernel pixels that have value 0 when convolving (only affects propagation of mask bits)

#define IGNORE_KERNEL_ZERO_PIXELS 0

namespace {
/*
 * Private functions to copy the border of an image
 *
 * copyRegion gets a bit complicated --- is it really worth it for private functions?
 */
/*
 * Copy a rectangular region from one Image to another
 */
template<typename OutImageT, typename InImageT>
inline void copyRegion(OutImageT &outImage,     // destination Image
                       InImageT const &inImage, // source Image
                       lsst::afw::image::Bbox const &region, // region to copy
                       int,
                       lsst::afw::image::detail::image_tag
                      ) {
    OutImageT outPatch(outImage, region); 
    InImageT inPatch(inImage, region);
    outPatch <<= OutImageT(inPatch, true);
}
// Specialization when the two types are the same
template<typename InImageT>
inline void copyRegion(InImageT &outImage,     // destination Image
                       InImageT const &inImage, // source Image
                       lsst::afw::image::Bbox const &region, // region to copy
                       int,
                       lsst::afw::image::detail::image_tag
                      ) {
    InImageT outPatch(outImage, region); 
    InImageT inPatch(inImage, region);
    outPatch <<= inPatch;
}

/*
 * Copy a rectangular region from one MaskedImage to another, setting the bits in orMask
 */
template<typename OutImageT, typename InImageT>
inline void copyRegion(OutImageT &outImage,     // destination Image
                       InImageT const &inImage, // source Image
                       lsst::afw::image::Bbox const &region, // region to copy
                       int orMask,                           // data to | into the mask pixels
                       lsst::afw::image::detail::maskedImage_tag
                      ) {
    OutImageT outPatch(outImage, region); 
    InImageT inPatch(inImage, region);
    outPatch <<= OutImageT(inPatch, true);
    *outPatch.getMask() |= orMask;
}
// Specialization when the two types are the same
template<typename InImageT>
inline void copyRegion(InImageT &outImage,      // destination Image
                       InImageT const &inImage, // source Image
                       lsst::afw::image::Bbox const &region, // region to copy
                       int orMask,                           // data to | into the mask pixels
                       lsst::afw::image::detail::maskedImage_tag
                      ) {
    InImageT outPatch(outImage, region);
    InImageT inPatch(inImage, region);
    outPatch <<= inPatch;
    *outPatch.getMask() |= orMask;
}


template <typename OutImageT, typename InImageT>
inline void copyBorder(
    OutImageT& convolvedImage,                           ///< convolved image
    InImageT const& inImage,                             ///< image to convolve
    lsst::afw::math::Kernel const &kernel,               ///< convolution kernel
    int edgeBit                         ///< mask bit to indicate border pixel;  if negative then no bit is set
) {
    const unsigned int imWidth = inImage.getWidth();
    const unsigned int imHeight = inImage.getHeight();
    const unsigned int kWidth = kernel.getWidth();
    const unsigned int kHeight = kernel.getHeight();
    const unsigned int kCtrX = kernel.getCtrX();
    const unsigned int kCtrY = kernel.getCtrY();
    
    using lsst::afw::image::Bbox;
    using lsst::afw::image::PointI;
    Bbox bottomEdge(PointI(0, 0), imWidth, kCtrY);
    copyRegion(convolvedImage, inImage, bottomEdge, edgeBit,
                typename lsst::afw::image::detail::image_traits<OutImageT>::image_category());
    
    int numHeight = kHeight - (1 + kCtrY);
    Bbox topEdge(PointI(0, imHeight - numHeight), imWidth, numHeight);
    copyRegion(convolvedImage, inImage, topEdge, edgeBit,
                typename lsst::afw::image::detail::image_traits<OutImageT>::image_category());
    
    Bbox leftEdge(PointI(0, kCtrY), kCtrX, imHeight + 1 - kHeight);
    copyRegion(convolvedImage, inImage, leftEdge, edgeBit,
                typename lsst::afw::image::detail::image_traits<OutImageT>::image_category());
    
    int numWidth = kWidth - (1 + kCtrX);
    Bbox rightEdge(PointI(imWidth - numWidth, kCtrY), numWidth, imHeight + 1 - kHeight);
    copyRegion(convolvedImage, inImage, rightEdge, edgeBit,
                typename lsst::afw::image::detail::image_traits<OutImageT>::image_category());
}
}

/**
 * @brief Apply convolution kernel to an image at one point
 *
 * @note: this is a high performance routine; the user is expected to:
 * - figure out the kernel center and adjust the supplied pixel accessors accordingly
 * For an example of how to do this see the convolve function.
 *
 * @ingroup afw
 */
template <typename OutImageT, typename InImageT>
inline typename OutImageT::Pixel::Constant lsst::afw::math::apply(
    typename InImageT::const_xy_locator& imageLocator,
                                        ///< locator for image pixel that overlaps (0,0) pixel of kernel(!)
    lsst::afw::image::Image<lsst::afw::math::Kernel::PixelT>::const_xy_locator &kernelLocator,
                                        ///< locator for (0,0) pixel of kernel
    int kWidth,                         ///< number of columns in kernel
    int kHeight                         ///< number of rows in kernel
                                  ) {
    typename OutImageT::Pixel::Constant outValue = 0;
    for (int y = 0; y != kHeight; ++y) {
        for (int x = 0; x != kWidth; ++x, ++imageLocator.x(), ++kernelLocator.x()) {
            typename lsst::afw::math::Kernel::Pixel::type const kVal = kernelLocator[0];
#if IGNORE_KERNEL_ZERO_PIXELS
            if (kVal != 0)
#endif
            {
                outValue = outValue + *imageLocator*InImageT::PixelCast(kVal);
            }
        }

        imageLocator  += lsst::afw::image::detail::difference_type(-kWidth, 1);
        kernelLocator += lsst::afw::image::detail::difference_type(-kWidth, 1);
    }

    imageLocator  += lsst::afw::image::detail::difference_type(0, -kHeight);
    kernelLocator += lsst::afw::image::detail::difference_type(0, -kHeight);

    return outValue;
}

/**
 * @brief Apply separable convolution kernel to an image at one point
 *
 * @note: this is a high performance routine; the user is expected to:
 * - figure out the kernel center and adjust the supplied pixel accessors accordingly
 * For an example of how to do this see the convolve function.
 *
 * @ingroup afw
 */
template <typename OutImageT, typename InImageT>
inline typename OutImageT::Pixel::Constant lsst::afw::math::apply(
    typename InImageT::const_xy_locator& imageLocator,
                                        ///< locator for image pixel that overlaps (0,0) pixel of kernel(!)
    std::vector<lsst::afw::math::Kernel::PixelT> const &kernelXList,  ///< kernel column vector
    std::vector<lsst::afw::math::Kernel::PixelT> const &kernelYList   ///< kernel row vector
) {
    typedef typename std::vector<lsst::afw::math::Kernel::PixelT>::const_iterator k_iter;

    std::vector<lsst::afw::math::Kernel::PixelT>::const_iterator kernelYIter = kernelYList.begin();

    typedef typename OutImageT::Pixel::Constant OutT;
    OutT outValue = 0;
    for (k_iter kernelYIter = kernelYList.begin(), end = kernelYList.end();
         kernelYIter != end; ++kernelYIter) {

        OutT outValueY = 0;
        for (k_iter kernelXIter = kernelXList.begin(), end = kernelXList.end();
             kernelXIter != end; ++kernelXIter, ++imageLocator.x()) {
            typename lsst::afw::math::Kernel::Pixel::type const kValX = *kernelXIter;
#if IGNORE_KERNEL_ZERO_PIXELS
            if (kValX != 0)
#endif
            {
                outValueY = outValueY + *imageLocator*InImageT::PixelCast(kValX);
            }
        }
        
        double const kValY = *kernelYIter;
#if IGNORE_KERNEL_ZERO_PIXELS
        if (kValY != 0)
#endif
        {
            outValue = outValue + outValueY*InImageT::PixelCast(kValY);
        }
        
        imageLocator += lsst::afw::image::detail::difference_type(-kernelXList.size(), 1);
    }
    
    imageLocator += lsst::afw::image::detail::difference_type(0, -kernelYList.size());

    return outValue;
}

/**
 * @brief Low-level convolution function that does not set edge pixels.
 *
 * convolvedImage must be the same size as inImage.
 * convolvedImage has a border in which the output pixels are not set. This border has size:
 * * kernel.getCtrX/Y() along the left/bottom edge
 * * kernel.getWidth/Height() - 1 - kernel.getCtrX/Y() along the right/top edge
 *
 * @throw lsst::pex::exceptions::InvalidParameter if convolvedImage is not the same size as inImage.
 * @throw lsst::pex::exceptions::InvalidParameter if inImage is smaller (in colums or rows) than kernel.
 *
 * @ingroup afw
 */
template <typename OutImageT, typename InImageT>
void lsst::afw::math::basicConvolve(
    OutImageT &convolvedImage,                  ///< convolved image
    InImageT const& inImage,                    ///< image to convolve
    lsst::afw::math::Kernel const& kernel,      ///< convolution kernel
    bool doNormalize                            ///< if True, normalize the kernel, else use "as is"
) {
    typedef typename lsst::afw::math::Kernel::PixelT KernelPixelT;
    typedef lsst::afw::image::Image<KernelPixelT> KernelImageT;

    typedef typename KernelImageT::const_xy_locator kXY_locator;
    typedef typename InImageT::const_xy_locator inXY_locator;
    typedef typename OutImageT::x_iterator cnvX_iterator;

    if (convolvedImage.dimensions() != inImage.dimensions()) {
        throw lsst::pex::exceptions::InvalidParameter("convolvedImage not the same size as inImage");
    }
    if (inImage.dimensions() < kernel.dimensions()) {
        throw lsst::pex::exceptions::InvalidParameter("inImage smaller than kernel in columns and/or rows");
    }
    
    int const inImageWidth = inImage.getWidth();
    int const inImageHeight = inImage.getHeight();
    int const kWidth = kernel.getWidth();
    int const kHeight = kernel.getHeight();
    int const cnvWidth = inImageWidth + 1 - kernel.getWidth();
    int const cnvHeight = inImageHeight + 1 - kernel.getHeight();
    int const cnvStartX = kernel.getCtrX();
    int const cnvStartY = kernel.getCtrY();
    int const cnvEndX = cnvStartX + cnvWidth;  // end index + 1
    int const cnvEndY = cnvStartY + cnvHeight; // end index + 1

    if (kernel.isSpatiallyVarying()) {
        lsst::pex::logging::Trace("lsst.afw.kernel.convolve", 3, "kernel is spatially varying");

        KernelImageT kernelImage(kernel.dimensions()); // the kernel at a point

        for (int cnvY = cnvStartY; cnvY != cnvEndY; ++cnvY) {
            double const rowPos = lsst::afw::image::indexToPosition(cnvY);
            
            inXY_locator  inImLoc =  inImage.xy_at(0, cnvY - cnvStartY);
            cnvX_iterator cnvImIter = convolvedImage.x_at(cnvStartX, cnvY);
            for (int cnvX = cnvStartX; cnvX != cnvEndX; ++cnvX, ++inImLoc.x(), ++cnvImIter) {
                double const colPos = lsst::afw::image::indexToPosition(cnvX);

                KernelPixelT kSum = kernel.computeImage(kernelImage, false, colPos, rowPos);
                kXY_locator kernelLoc = kernelImage.xy_at(0,0);
                *cnvImIter = lsst::afw::math::apply<OutImageT, InImageT>(inImLoc, kernelLoc, kWidth, kHeight);
                if (doNormalize) {
                    *cnvImIter = *cnvImIter/kSum;
                }
            }
        }
    } else {                            // kernel is spatially invariant
        lsst::pex::logging::Trace("lsst.afw.kernel.convolve", 3, "kernel is spatially invariant");
        KernelImageT kernelImage(kernel.dimensions());
        (void)kernel.computeImage(kernelImage, doNormalize);

        for (int cnvY = cnvStartY; cnvY != cnvEndY; ++cnvY) {
            inXY_locator inImLoc =  inImage.xy_at(0, cnvY - cnvStartY);
            for (cnvX_iterator cnvImIter = convolvedImage.x_at(cnvStartX, cnvY),
                     cnvImEnd = convolvedImage.x_at(cnvEndX, cnvY); cnvImIter != cnvImEnd; ++inImLoc.x(), ++cnvImIter) {
                kXY_locator kernelLoc = kernelImage.xy_at(0,0);
                *cnvImIter = lsst::afw::math::apply<OutImageT, InImageT>(inImLoc, kernelLoc, kWidth, kHeight);
            }
        }
    }
}

/************************************************************************************************************/
/**
 * @brief A version of basicConvolve that should be used when convolving delta function kernels
 */
template <typename OutImageT, typename InImageT>
void lsst::afw::math::basicConvolve(
    OutImageT& convolvedImage,            ///< convolved image
    InImageT const& inImage,          ///< image to convolve
    lsst::afw::math::DeltaFunctionKernel const &kernel,    ///< convolution kernel
    bool doNormalize    ///< if True, normalize the kernel, else use "as is"
) {
    assert (!kernel.isSpatiallyVarying());

    if (convolvedImage.dimensions() != inImage.dimensions()) {
        throw lsst::pex::exceptions::InvalidParameter("convolvedImage not the same size as inImage");
    }
    if (convolvedImage.dimensions() < kernel.dimensions()) {
        throw lsst::pex::exceptions::InvalidParameter(
            "inImage smaller than kernel in columns and/or rows");
    }
    
    int const mImageWidth = inImage.getWidth(); // size of input region
    int const mImageHeight = inImage.getHeight();
    int const cnvWidth = mImageWidth + 1 - kernel.getWidth();
    int const cnvHeight = mImageHeight + 1 - kernel.getHeight();
    int const cnvStartX = kernel.getCtrX();
    int const cnvStartY = kernel.getCtrY();
    int const inStartX = kernel.getPixel().first;
    int const inStartY = kernel.getPixel().second;

    lsst::pex::logging::Trace("lsst.afw.kernel.convolve", 3, "kernel is a spatially invariant delta function basis");

    for (int i = 0; i < cnvHeight; ++i) {
        typename InImageT::x_iterator
            inPtr =         inImage.row_begin(i +  inStartY) +  inStartX;
        typename OutImageT::x_iterator
            cnvPtr = convolvedImage.row_begin(i + cnvStartY) + cnvStartX,
            cnvEnd = cnvPtr + cnvWidth;

        for (; cnvPtr != cnvEnd; ++cnvPtr, ++inPtr) {
            *cnvPtr = OutImageT::PixelCast(*inPtr);
        }
    }
}

/**
 * @brief A version of basicConvolve that should be used when convolving separable kernels
 */
template <typename OutImageT, typename InImageT>
void lsst::afw::math::basicConvolve(
    OutImageT& convolvedImage,        ///< convolved image
    InImageT const& inImage,          ///< image to convolve
    lsst::afw::math::SeparableKernel const &kernel,  ///< convolution kernel
    bool doNormalize                                 ///< if True, normalize the kernel, else use "as is"
) {
    typedef typename lsst::afw::math::Kernel::PixelT KernelPixelT;

    typedef typename InImageT::const_xy_locator inXY_locator;
    typedef typename OutImageT::x_iterator cnvX_iterator;

    if (convolvedImage.dimensions() != inImage.dimensions()) {
        throw lsst::pex::exceptions::InvalidParameter("convolvedImage not the same size as inImage");
    }
    if (inImage.dimensions() < kernel.dimensions()) {
        throw lsst::pex::exceptions::InvalidParameter(
            "inImage smaller than kernel in columns and/or rows");
    }
    
    int const imWidth = inImage.getWidth();
    int const imHeight = inImage.getHeight();
    int const cnvWidth = static_cast<int>(imWidth) + 1 - static_cast<int>(kernel.getWidth());
    int const cnvHeight = static_cast<int>(imHeight) + 1 - static_cast<int>(kernel.getHeight());
    int const cnvStartX = static_cast<int>(kernel.getCtrX());
    int const cnvStartY = static_cast<int>(kernel.getCtrY());
    int const cnvEndX = cnvStartX + static_cast<int>(cnvWidth); // end index + 1
    int const cnvEndY = cnvStartY + static_cast<int>(cnvHeight); // end index + 1

    std::vector<lsst::afw::math::Kernel::PixelT> kXVec(kernel.getWidth());
    std::vector<lsst::afw::math::Kernel::PixelT> kYVec(kernel.getHeight());
    
    if (kernel.isSpatiallyVarying()) {
        lsst::pex::logging::Trace("lsst.afw.kernel.convolve", 3, "kernel is a spatially varying separable kernel");

        for (int cnvY = cnvStartY; cnvY != cnvEndY; ++cnvY) {
            double const rowPos = lsst::afw::image::indexToPosition(cnvY);
            
            inXY_locator  inImLoc =  inImage.xy_at(0, cnvY - cnvStartY);
            cnvX_iterator cnvImIter = convolvedImage.row_begin(cnvY) + cnvStartX;
            for (int cnvX = cnvStartX; cnvX != cnvEndX; ++cnvX, ++inImLoc.x(), ++cnvImIter) {
                double const colPos = lsst::afw::image::indexToPosition(cnvX);

                KernelPixelT kSum = kernel.computeVectors(kXVec, kYVec, doNormalize, colPos, rowPos);

                *cnvImIter = lsst::afw::math::apply<OutImageT, InImageT>(inImLoc, kXVec, kYVec);
                if (doNormalize) {
                    *cnvImIter = *cnvImIter/kSum;
                }
            }
        }
    } else {
        // kernel is spatially invariant
        lsst::pex::logging::Trace("lsst.afw.kernel.convolve", 3, "kernel is a spatially invariant separable kernel");

        (void)kernel.computeVectors(kXVec, kYVec, doNormalize);

        for (int cnvY = cnvStartY; cnvY != cnvEndY; ++cnvY) {
            inXY_locator inImLoc =  inImage.xy_at(0, cnvY - cnvStartY);
            for (cnvX_iterator cnvImIter = convolvedImage.row_begin(cnvY) + cnvStartX,
                     cnvImEnd = cnvImIter + cnvEndX - cnvStartX; cnvImIter != cnvImEnd; ++inImLoc.x(), ++cnvImIter) {
                *cnvImIter = lsst::afw::math::apply<OutImageT, InImageT>(inImLoc, kXVec, kYVec);
            }
        }
    }
}

/**
 * @brief Convolve an Image with a Kernel, setting pixels of an existing image
 *
 * convolvedImage must be the same size as inImage.
 * convolvedImage has a border in which the output pixels are just a copy of the input pixels.
 * This border has size:
 * * kernel.getCtrX/Y() along the left/bottom edge
 * * kernel.getWidth/Height() - 1 - kernel.getCtrY/Y() along the right/top edge
 *
 * @throw lsst::pex::exceptions::InvalidParameter if convolvedImage is not the same size as inImage.
 * @throw lsst::pex::exceptions::InvalidParameter if inImage is smaller (in colums or rows) than kernel.
 *
 * @ingroup afw
 */
template <typename OutImageT, typename InImageT, typename KernelT>
void lsst::afw::math::convolve(
    OutImageT& convolvedImage,          ///< convolved image
    InImageT const& inImage,            ///< image to convolve
    KernelT const& kernel,              ///< convolution kernel
    bool doNormalize,                   ///< if True, normalize the kernel, else use "as is"
    int edgeBit                         ///< mask bit to indicate pixel includes edge-extended data;
                                        ///< if negative (default) then no bit is set; only relevant for MaskedImages
) {
    lsst::afw::math::basicConvolve(convolvedImage, inImage, kernel, doNormalize);
    copyBorder(convolvedImage, inImage, kernel, edgeBit);
}

/**
 * @brief Convolve an Image with a LinearCombinationKernel, setting pixels of an existing image.
 *
 * A variant of the convolve function that is faster for spatially varying LinearCombinationKernels.
 * For the sake of speed the kernel is NOT normalized. If you want normalization then call the standard
 * convolve function.
 *
 * The Algorithm:
 * Convolves the input Image by each basis kernel in turn, creating a set of basis images.
 * Then for each output pixel it solves the spatial model and computes the the pixel as
 * the appropriate linear combination of basis images.
 *
 * @throw lsst::pex::exceptions::InvalidParameter if convolvedImage is not the same size as inImage.
 * @throw lsst::pex::exceptions::InvalidParameter if inImage is smaller (in colums or rows) than kernel.
 *
 * @ingroup afw
 */
template <typename OutImageT, typename InImageT>
void lsst::afw::math::convolveLinear(
    OutImageT& convolvedImage,          ///< convolved image
    InImageT const& inImage,            ///< image to convolve
    lsst::afw::math::LinearCombinationKernel const& kernel, ///< convolution kernel
    int edgeBit				///< mask bit to indicate pixel includes edge-extended data;
                                        ///< if negative (default) then no bit is set; only relevant for MaskedImages
) {
    if (!kernel.isSpatiallyVarying()) {
        return lsst::afw::math::convolve(convolvedImage, inImage, kernel, false, edgeBit);
    }

    if (convolvedImage.dimensions() != inImage.dimensions()) {
        throw lsst::pex::exceptions::InvalidParameter("convolvedImage not the same size as inImage");
    }
    if (inImage.dimensions() < kernel.dimensions()) {
        throw lsst::pex::exceptions::InvalidParameter("inImage smaller than kernel in columns and/or rows");
    }
    
    typedef typename InImageT::template ImageTypeFactory<double>::type BasisImage;
    typedef typename BasisImage::Ptr BasisImagePtr;
    typedef typename BasisImage::x_iterator BasisX_iterator;
    typedef std::vector<BasisX_iterator> BasisX_iteratorList;
    typedef typename OutImageT::x_iterator cnvX_iterator;
    typedef lsst::afw::math::LinearCombinationKernel::KernelList kernelListType;

    int const imWidth = inImage.getWidth();
    int const imHeight = inImage.getHeight();
    int const cnvWidth = imWidth + 1 - kernel.getWidth();
    int const cnvHeight = imHeight + 1 - kernel.getHeight();
    int const cnvStartX = kernel.getCtrX();
    int const cnvStartY = kernel.getCtrY();
    int const cnvEndX = cnvStartX + cnvWidth;  // end index + 1
    int const cnvEndY = cnvStartY + cnvHeight; // end index + 1
    // create a vector of pointers to basis images (input Image convolved with each basis kernel)
    std::vector<BasisImagePtr> basisImagePtrList;
    {
        kernelListType basisKernelList = kernel.getKernelList();
        for (typename kernelListType::const_iterator basisKernelIter = basisKernelList.begin();
             basisKernelIter != basisKernelList.end(); ++basisKernelIter) {
            BasisImagePtr basisImagePtr(new BasisImage(inImage.dimensions()));
            lsst::afw::math::basicConvolve(*basisImagePtr, inImage, **basisKernelIter, false);
            basisImagePtrList.push_back(basisImagePtr);
        }
    }
    BasisX_iteratorList basisIterList;                           // x_iterators for each basis image
    basisIterList.reserve(basisImagePtrList.size());
    
    // iterate over matching pixels of all images to compute output image
    std::vector<double> kernelCoeffList(kernel.getNKernelParameters()); // weights of basic images at this point
    for (int cnvY = cnvStartY; cnvY < cnvEndY; ++cnvY) {
        double const rowPos = lsst::afw::image::indexToPosition(cnvY);
    
        cnvX_iterator cnvXIter = convolvedImage.row_begin(cnvY) + cnvStartX;
        for (unsigned int i = 0; i != basisIterList.size(); ++i) {
            basisIterList[i] = basisImagePtrList[i]->row_begin(cnvY) + cnvStartX;
        }

        for (int cnvX = cnvStartX; cnvX != cnvEndX; ++cnvX, ++cnvXIter) {
            double const colPos = lsst::afw::image::indexToPosition(cnvX);
            
            kernel.computeKernelParametersFromSpatialModel(kernelCoeffList, colPos, rowPos);

            typename OutImageT::Pixel::Constant cnvImagePix = 0.0;
            for (unsigned int i = 0; i != basisIterList.size(); ++i) {
                cnvImagePix = cnvImagePix + kernelCoeffList[i]*(*basisIterList[i]);
                ++basisIterList[i];
            }
            *cnvXIter = cnvImagePix;
        }
    }

    copyBorder(convolvedImage, inImage, kernel, edgeBit);
}
