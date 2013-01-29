// -*- lsst-++ -*-

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

%include "lsst/afw/cameraGeom/cameraGeom_fwd.i"

%lsst_exceptions();

%{
#include "lsst/afw/image/Image.h"
#include "lsst/afw/image/MaskedImage.h"
#include "lsst/afw/fits.h"
%}

%import  "lsst/afw/image/image.i"
%import  "lsst/afw/image/maskedImage.i"

%include "lsst/afw/cameraGeom/Id.i"
%include "lsst/afw/cameraGeom/FpPoint.i"
%include "lsst/afw/cameraGeom/Orientation.i"
%include "lsst/afw/cameraGeom/Detector.i"

%include "lsst/afw/cameraGeom/Amp.i"
%include "lsst/afw/cameraGeom/DetectorMosaic.i"
%include "lsst/afw/cameraGeom/Ccd.i"
%include "lsst/afw/cameraGeom/Raft.i"
%include "lsst/afw/cameraGeom/Camera.i"
%include "lsst/afw/cameraGeom/Distortion.i"
