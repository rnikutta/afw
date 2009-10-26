// -*- LSST-C++ -*-
/**
 * @file
 *
 * @brief Definitions of DeltaFunctionKernel member functions.
 *
 * @ingroup fw
 */
#include <vector>

#include "lsst/pex/exceptions.h"
#include "lsst/afw/math/Kernel.h"

namespace pexExcept = lsst::pex::exceptions;
namespace afwMath = lsst::afw::math;
namespace afwImage = lsst::afw::image;

/**
 * @brief Construct a spatially invariant DeltaFunctionKernel
 *
 * @throw pexExcept::InvalidParameterException if active pixel is off the kernel
 */
afwMath::DeltaFunctionKernel::DeltaFunctionKernel(
    int width,              ///< kernel size (columns)
    int height,             ///< kernel size (rows)
    afwImage::PointI const &point   ///< index of active pixel (where 0,0 is the lower left corner)
) :
    Kernel(width, height, 0),
    _pixel(point)
{
    if (point.getX() < 0 || point.getX() >= width || point.getY() < 0 || point.getY() >= height) {
        throw LSST_EXCEPT(pexExcept::InvalidParameterException, "Active pixel lies outside image");
    }
}

afwMath::Kernel::Ptr afwMath::DeltaFunctionKernel::clone() const {
    afwMath::Kernel::Ptr retPtr(new afwMath::DeltaFunctionKernel(this->getWidth(), this->getHeight(),
        this->_pixel));
    retPtr->setCtrX(this->getCtrX());
    retPtr->setCtrY(this->getCtrY());
    return retPtr;
}

double afwMath::DeltaFunctionKernel::computeImage(
    afwImage::Image<Pixel> &image,
    bool doNormalize,
    double x,
    double y
) const {
    if (image.getDimensions() != this->getDimensions()) {
        throw LSST_EXCEPT(pexExcept::InvalidParameterException, "image is the wrong size");
    }

    const int pixelX = getPixel().getX(); // active pixel in Kernel
    const int pixelY = getPixel().getY();

    image = 0;
    *image.xy_at(pixelX, pixelY) = 1;

    return 1;
}

std::string afwMath::DeltaFunctionKernel::toString(std::string prefix) const {
    const int pixelX = getPixel().getX(); // active pixel in Kernel
    const int pixelY = getPixel().getY();

    std::ostringstream os;            
    os << prefix << "DeltaFunctionKernel:" << std::endl;
    os << prefix << "Pixel (c,r) " << pixelX << "," << pixelY << ")" << std::endl;
    os << Kernel::toString(prefix + "\t");
    return os.str();
};
