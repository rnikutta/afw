// -*- lsst-c++ -*-

/* 
 * LSST Data Management System
 * Copyright 2008, 2009, 2010 LSST Corporation.
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
 
%{
#include "boost/shared_ptr.hpp"
#include "lsst/afw/detection/Threshold.h"
#include "lsst/afw/detection/Footprint.h"
#include "lsst/afw/detection/FootprintMerge.h"
#include "lsst/afw/detection/FootprintCtrl.h"
#include "lsst/afw/detection/HeavyFootprint.h"
#include "lsst/afw/detection/FootprintFunctor.h"
#include "lsst/afw/detection/FootprintArray.h"
#include "lsst/afw/detection/FootprintArray.cc"
#include "ndarray.h"
%}

namespace lsst { namespace afw { namespace detection {
typedef lsst::afw::geom::Span Span;
}}}

%import "lsst/afw/table/io/ioLib.i"

%include "lsst/afw/image/LsstImageTypes.h"
%include "std_pair.i"
%template(pairBB) std::pair<bool, bool>;

%shared_vec(boost::shared_ptr<lsst::afw::detection::Footprint>);

%ignore lsst::afw::detection::FootprintFunctor::operator();

// already in image.i.
// %template(VectorBox2I) std::vector<lsst::afw::geom::Box2I>;

%declareTablePersistable(Footprint, lsst::afw::detection::Footprint);
%shared_ptr(std::vector<boost::shared_ptr<lsst::afw::detection::Footprint> >);

%declareNumPyConverters(ndarray::Array<unsigned short,1,1>);
%declareNumPyConverters(ndarray::Array<float,1,1>);

%define %HeavyFootprintPtr(NAME, PIXEL_TYPE, MASK_TYPE, VAR_TYPE)
%declareTablePersistable(HeavyFootprint ## NAME, lsst::afw::detection::HeavyFootprint<PIXEL_TYPE, MASK_TYPE, VAR_TYPE>);
%declareNumPyConverters(ndarray::Array<PIXEL_TYPE,1,1>);
%enddef

%HeavyFootprintPtr(I, int,   lsst::afw::image::MaskPixel, lsst::afw::image::VariancePixel)
%HeavyFootprintPtr(F, float, lsst::afw::image::MaskPixel, lsst::afw::image::VariancePixel)

%rename(assign) lsst::afw::detection::Footprint::operator=;

%template(SpanContainerT)      std::vector<boost::shared_ptr<lsst::afw::geom::Span> >;
%template(FootprintList)       std::vector<boost::shared_ptr<lsst::afw::detection::Footprint> >;

%include "lsst/afw/detection/Threshold.h"
%include "lsst/afw/detection/Footprint.h"
%include "lsst/afw/detection/FootprintMerge.h"
%include "lsst/afw/detection/FootprintCtrl.h"
%include "lsst/afw/detection/HeavyFootprint.h"
%include "lsst/afw/detection/FootprintFunctor.h"

%define %thresholdOperations(TYPE)
    %extend lsst::afw::detection::Threshold {
        %template(getValue) getValue<TYPE<unsigned short> >;
        %template(getValue) getValue<TYPE<int> >;
        %template(getValue) getValue<TYPE<float> >;
        %template(getValue) getValue<TYPE<double> >;
    }
%enddef

%define %footprintOperations(PIXEL)
%template(insertIntoImage) lsst::afw::detection::Footprint::insertIntoImage<PIXEL>;
%enddef

%extend lsst::afw::detection::Footprint {
    %template(intersectMask) intersectMask<lsst::afw::image::MaskPixel>;
    %footprintOperations(unsigned short)
    %footprintOperations(int)
    %footprintOperations(boost::uint64_t)
}

%define %heavyFootprints(NAME, PIXEL_TYPE, MASK_TYPE, VAR_TYPE)
    %template(HeavyFootprint ##NAME) lsst::afw::detection::HeavyFootprint<PIXEL_TYPE, MASK_TYPE, VAR_TYPE>;

/*
%extend lsst::afw::detection::HeavyFootprint<PIXEL_TYPE, MASK_TYPE, VAR_TYPE> {
    ndarray::Array<PIXEL_TYPE,1,1> getImageArray() { self->getImageArray(); }
    ndarray::Array<MASK_TYPE,1,1> getMaskArray() { self->getMaskArray(); }
    ndarray::Array<VAR_TYPE,1,1> getVarianceArray() { self->getVarianceArray(); }
}
 */

    %template(HeavyFootprintPtrList ## NAME) std::vector<boost::shared_ptr<lsst::afw::detection::HeavyFootprint<PIXEL_TYPE, MASK_TYPE, VAR_TYPE> > >;


    %template(makeHeavyFootprint ##NAME) lsst::afw::detection::makeHeavyFootprint<PIXEL_TYPE, MASK_TYPE, VAR_TYPE>;

    %template(mergeHeavyFootprints ##NAME) lsst::afw::detection::mergeHeavyFootprints<PIXEL_TYPE, MASK_TYPE, VAR_TYPE>;

    %inline %{
        //------------------------------------------------------------------------
        // THIS CAST INTERFACE IS DEPRECATED IN FAVOR OF %castShared AND WILL BE REMOVED IN THE FUTURE
        PTR(lsst::afw::detection::HeavyFootprint<PIXEL_TYPE, MASK_TYPE, VAR_TYPE>)
            /**
             * Cast a Footprint to a HeavyFootprint of a specified type
             */
            cast_HeavyFootprint##NAME(PTR(lsst::afw::detection::Footprint) foot) {
            return boost::dynamic_pointer_cast<lsst::afw::detection::HeavyFootprint<PIXEL_TYPE, MASK_TYPE, VAR_TYPE> >(foot);
        }
        //------------------------------------------------------------------------
        PTR(lsst::afw::detection::HeavyFootprint<PIXEL_TYPE, MASK_TYPE, VAR_TYPE>)
            /**
             * Cast a Footprint to a HeavyFootprint; the MaskedImage disambiguates the type
             */
            cast_HeavyFootprint(PTR(lsst::afw::detection::Footprint) foot,
                                lsst::afw::image::MaskedImage<PIXEL_TYPE, MASK_TYPE, VAR_TYPE> const&) {
            return boost::dynamic_pointer_cast<lsst::afw::detection::HeavyFootprint<PIXEL_TYPE, MASK_TYPE, VAR_TYPE> >(foot);
        }
    %}
#define DERIVED_ARGS PIXEL_TYPE, MASK_TYPE, VAR_TYPE
    %castShared(lsst::afw::detection::HeavyFootprint< DERIVED_ARGS >,
                lsst::afw::detection::Footprint)
#undef DERIVED_ARGS

%enddef



%define %footprintImageOperations(PIXEL_TYPE)
    %template(setImageFromFootprint) lsst::afw::detection::setImageFromFootprint<lsst::afw::image::Image<PIXEL_TYPE> >;
    %template(setImageFromFootprintList)
    lsst::afw::detection::setImageFromFootprintList<lsst::afw::image::Image<PIXEL_TYPE> >;
    %template(copyWithinFootprintImage)
    lsst::afw::detection::copyWithinFootprint<lsst::afw::image::Image<PIXEL_TYPE> >;
    %template(copyWithinFootprintMaskedImage)
    lsst::afw::detection::copyWithinFootprint<lsst::afw::image::MaskedImage<PIXEL_TYPE> >;
    %template(clipToNonzero) lsst::afw::detection::Footprint::clipToNonzero<PIXEL_TYPE>;
%enddef

%define %imageOperations(NAME, PIXEL_TYPE)
    %template(FootprintFunctor ##NAME) lsst::afw::detection::FootprintFunctor<lsst::afw::image::Image<PIXEL_TYPE> >;
    %template(FootprintFunctorMI ##NAME)
                       lsst::afw::detection::FootprintFunctor<lsst::afw::image::MaskedImage<PIXEL_TYPE> >;
    %footprintImageOperations(PIXEL_TYPE);
%enddef

%define %maskOperations(PIXEL_TYPE)
    %template(footprintAndMask) lsst::afw::detection::footprintAndMask<PIXEL_TYPE>;
    %template(setMaskFromFootprint) lsst::afw::detection::setMaskFromFootprint<PIXEL_TYPE>;
    %template(clearMaskFromFootprint) lsst::afw::detection::clearMaskFromFootprint<PIXEL_TYPE>;
    %template(setMaskFromFootprintList) lsst::afw::detection::setMaskFromFootprintList<PIXEL_TYPE>;
%enddef

%heavyFootprints(I, int,   lsst::afw::image::MaskPixel, lsst::afw::image::VariancePixel)
%heavyFootprints(F, float, lsst::afw::image::MaskPixel, lsst::afw::image::VariancePixel)


%thresholdOperations(lsst::afw::image::Image);
%thresholdOperations(lsst::afw::image::MaskedImage);
%imageOperations(F, float);
%imageOperations(D, double);
%footprintImageOperations(boost::uint16_t);
%footprintImageOperations(int);
%footprintImageOperations(boost::uint64_t);
%maskOperations(lsst::afw::image::MaskPixel);
%template(FootprintFunctorMaskU) lsst::afw::detection::FootprintFunctor<lsst::afw::image::Mask<boost::uint16_t> >;

// because stupid SWIG's %template doesn't work on these functions
%define %footprintArrayTemplates(T)
%declareNumPyConverters(ndarray::Array<T,1,0>);
%declareNumPyConverters(ndarray::Array<T,2,0>);
%declareNumPyConverters(ndarray::Array<T,3,0>);
%declareNumPyConverters(ndarray::Array<T const,1,0>);
%declareNumPyConverters(ndarray::Array<T const,2,0>);
%declareNumPyConverters(ndarray::Array<T const,3,0>);
%inline %{
    void flattenArray(
        lsst::afw::detection::Footprint const & fp,
        ndarray::Array<T const,2,0> const & src,
        ndarray::Array<T,1,0> const & dest,
        lsst::afw::geom::Point2I const & origin = lsst::afw::geom::Point2I()
    ) {
        lsst::afw::detection::flattenArray(fp, src, dest, origin);
    }    
    void flattenArray(
        lsst::afw::detection::Footprint const & fp,
        ndarray::Array<T const,3,0> const & src,
        ndarray::Array<T,2,0> const & dest,
        lsst::afw::geom::Point2I const & origin = lsst::afw::geom::Point2I()
    ) {
        lsst::afw::detection::flattenArray(fp, src, dest, origin);
    }    
    void expandArray(
        lsst::afw::detection::Footprint const & fp,
        ndarray::Array<T const,1,0> const & src,
        ndarray::Array<T,2,0> const & dest,
        lsst::afw::geom::Point2I const & origin = lsst::afw::geom::Point2I()
    ) {
        lsst::afw::detection::expandArray(fp, src, dest, origin);
    }
    void expandArray(
        lsst::afw::detection::Footprint const & fp,
        ndarray::Array<T const,2,0> const & src,
        ndarray::Array<T,3,0> const & dest,
        lsst::afw::geom::Point2I const & origin = lsst::afw::geom::Point2I()
    ) {
        lsst::afw::detection::expandArray(fp, src, dest, origin);
    }
%}
%{
    template void lsst::afw::detection::flattenArray(
        lsst::afw::detection::Footprint const &,
        ndarray::Array<T const,2,0> const &,
        ndarray::Array<T,1,0> const &,
        lsst::afw::geom::Point2I const &
    );
    template void lsst::afw::detection::flattenArray(
        lsst::afw::detection::Footprint const &,
        ndarray::Array<T const,3,0> const &,
        ndarray::Array<T,2,0> const &,
        lsst::afw::geom::Point2I const &
    );
    template void lsst::afw::detection::expandArray(
        lsst::afw::detection::Footprint const &,
        ndarray::Array<T const,1,0> const &,
        ndarray::Array<T,2,0> const &,
        lsst::afw::geom::Point2I const &
    );
    template void lsst::afw::detection::expandArray(
        lsst::afw::detection::Footprint const &,
        ndarray::Array<T const,2,0> const &,
        ndarray::Array<T,3,0> const &,
        lsst::afw::geom::Point2I const &
    );
%}
%enddef

%footprintArrayTemplates(boost::uint16_t);
%footprintArrayTemplates(int);
%footprintArrayTemplates(float);
%footprintArrayTemplates(double);


%pythoncode %{
makeHeavyFootprint = makeHeavyFootprintF
%}

%extend lsst::afw::detection::FootprintMergeList {
%pythoncode %{

        def getMergedSourceCatalog(self, catalogs, filters,
                                   peakDist, schema, idFactory, samePeakDist):
            """Add multiple catalogs and get the SourceCatalog with merged Footprints"""
            import lsst.afw.table as afwTable

            table = afwTable.SourceTable.make(schema, idFactory)
            mergedList = afwTable.SourceCatalog(table)

            # if peak is not an array, create an array the size of catalogs
            try:
                len(samePeakDist)
            except:
                samePeakDist = [samePeakDist] * len(catalogs)

            try:
                len(peakDist)
            except:
                peakDist = [peakDist] * len(catalogs)

            if len(peakDist) != len(catalogs):
                raise ValueError("Number of catalogs (%d) does not match length of peakDist (%d)"
                                 % (len(catalogs), len(peakDist)))

            if len(samePeakDist) != len(catalogs):
                raise ValueError("Number of catalogs (%d) does not match length of samePeakDist (%d)"
                                 % (len(catalogs), len(samePeakDist)))

            if len(filters) != len(catalogs):
                raise ValueError("Number of catalogs (%d) does not match number of filters (%d)"
                                 % (len(catalogs), len(filters)))

            self.clearCatalog()
            for cat, filter, dist, sameDist in zip(catalogs, filters, peakDist, samePeakDist):
                self.addCatalog(table, cat, filter, dist, True, sameDist)

            self.getFinalSources(mergedList)
            return mergedList
%}
}
