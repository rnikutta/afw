// -*- lsst-C++ -*-

%{
#include "lsst/afw/detection/Psf.h"
%}

%shared_ptr(lsst::afw::detection::Psf);
%shared_ptr(lsst::afw::detection::KernelPsf);

%ignore PsfFactoryBase;
%ignore lsst::afw::detection::Psf::writeToRecords;
%ignore lsst::afw::detection::Psf::readFromRecords;
%include "lsst/afw/detection/Psf.h"

%lsst_persistable(lsst::afw::detection::Psf);
