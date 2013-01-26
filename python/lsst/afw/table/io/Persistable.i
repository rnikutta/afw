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

%module(package="lsst.afw.table.io") ioLib

#pragma SWIG nowarn=362                 // operator=  ignored
#pragma SWIG nowarn=389                 // operator[]  ignored
#pragma SWIG nowarn=503                 // comparison operators ignored

%include "boost_shared_ptr.i"
%include "lsst/p_lsstSwig.i"
%import "lsst/pex/exceptions/exceptionsLib.i"

%lsst_exceptions();

%{
#include "lsst/afw/table/io/Persistable.h"
%}

%shared_ptr(lsst::afw::table::io::Persistable);

%define %declareTablePersistable(NAME, T)
%shared_ptr(lsst::afw::table::io::Persistable);
%shared_ptr(lsst::afw::table::io::PersistableFacade< T >);
%shared_ptr(T);
%template(NAME ## PersistableFacade) lsst::afw::table::io::PersistableFacade< T >;
%enddef

%include "lsst/afw/table/io/Persistable.h"
