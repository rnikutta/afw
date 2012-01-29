// -*- lsst-c++ -*-
#ifndef LSST_AFW_fits_h_INCLUDED
#define LSST_AFW_fits_h_INCLUDED

/**
 *  @file lsst/afw/fits.h
 *
 *  Utilities for working with FITS files.
 *  This was written as part of implementing the afw/table library.  Someday
 *  the afw/image FITS I/O should be modified to use some of these with the goal
 *  of sharing code between the two.
 */

#include <string>

#include <boost/format.hpp>

#include "lsst/pex/exceptions.h"

namespace lsst { namespace afw { namespace fits {

class HeaderIterationFunctor {
public:

    virtual void operator()(char const * key, char const * value, char const * comment) = 0;

    virtual ~HeaderIterationFunctor() {}

};

/**
 * @brief An exception thrown when problems are found when reading or writing FITS files.
 */
LSST_EXCEPTION_TYPE(FitsError, lsst::pex::exceptions::Exception, lsst::afw::fits::FitsError)

/**
 * @brief An exception thrown when a FITS file has the wrong type.
 */
LSST_EXCEPTION_TYPE(FitsTypeError, lsst::afw::fits::FitsError, lsst::afw::fits::FitsTypeError)

//@{
/**
 *  @brief Return an error message reflecting FITS I/O errors.
 *
 *  @param[in] fileName   FITS filename to be included in the error message.
 *  @param[in] status     The last status value returned by the cfitsio library; if nonzero,
 *                        the error message will include a description from cfitsio.
 *  @param[in] msg        An additional custom message to include.
 */
std::string makeErrorMessage(std::string const & fileName="", int status=0, std::string const & msg="");
inline std::string makeErrorMessage(std::string const & fileName, int status, boost::format const & msg) {
    return makeErrorMessage(fileName, status, msg.str());
}
//@}

//@{
/**
 *  @brief Return an error message reflecting FITS I/O errors.
 *
 *  @param[in] fptr       A cfitsio fitsfile pointer to be inspected for a filename.
 *                        Passed as void* to avoid including fitsio.h in the header file.
 *  @param[in] status     The last status value returned by the cfitsio library; if nonzero,
 *                        the error message will include a description from cfitsio.
 *  @param[in] msg        An additional custom message to include.
 */
std::string makeErrorMessage(void * fptr, int status=0, std::string const & msg="");
inline std::string makeErrorMessage(void * fptr, int status, boost::format const & msg) {
    return makeErrorMessage(fptr, status, msg.str());
}
//@}

/**
 *  @brief A simple struct that combines the two arguments that must be passed to most cfitsio routines
 *         and contains thin and/or templated wrappers around common cfitsio routines.
 *
 *  This is NOT intended to be an object-oriented C++ wrapper around cfitsio; it's simply a thin layer that
 *  saves a lot of repetition and const/reinterpret casts.
 */
struct Fits {

    void updateKey(char const * key, char const * value, char const * comment=0);

    void writeKey(char const * key, char const * value, char const * comment=0);

    template <typename T>
    void updateKey(char const * key, T value, char const * comment=0);

    template <typename T>
    void writeKey(char const * key, T value, char const * comment=0);

    template <typename T>
    void updateColumnKey(char const * prefix, int n, T value, char const * comment=0);

    template <typename T>
    void writeColumnKey(char const * prefix, int n, T value, char const * comment=0);

    template <typename T>
    void readKey(char const * key, T & value);

    void readKey(char const * key, std::string & value);

    void forEachKey(HeaderIterationFunctor & functor);

    /**
     *  @brief Add a column to a table
     *
     *  If size <= 0, the field will be a variable length array, with max set by (-size),
     *  or left unknown if size == 0.
     */
    template <typename T>
    int addColumn(char const * ttype, int size, char const * comment=0);

    /// @brief Append rows to a table, and return the index of the first new row.
    std::size_t addRows(std::size_t nRows);

    std::size_t countRows();

    template <typename T>
    void writeTableArray(std::size_t row, int col, int nElements, T const * value);

    template <typename T>
    void writeTableScalar(std::size_t row, int col, T value) { writeTableArray(row, col, 1, &value); }

    template <typename T>
    void readTableArray(std::size_t row, int col, int nElements, T * value);

    template <typename T>
    void readTableScalar(std::size_t row, int col, T & value) { readTableArray(row, col, 1, &value); }
    
    long getTableArraySize(int col);

    long getTableArraySize(std::size_t row, int col);

    static Fits createFile(char const * filename);

    static Fits openFile(char const * filename, bool writeable);

    void createTable();

    void closeFile();

    void checkStatus() const {
        if (status != 0) throw LSST_EXCEPT(FitsError, makeErrorMessage(fptr, status));
    }

    void * fptr;
    int status;
}; 

}}} /// namespace lsst::afw::fits

#endif // !LSST_AFW_fits_h_INCLUDED