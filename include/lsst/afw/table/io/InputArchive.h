// -*- lsst-c++ -*-
#ifndef AFW_TABLE_IO_InputArchive_h_INCLUDED
#define AFW_TABLE_IO_InputArchive_h_INCLUDED

#include <vector>
#include <map>

#include "lsst/base.h"
#include "lsst/afw/table/Catalog.h"

namespace lsst { namespace afw { 

namespace fits {

class Fits;

} // namespace fits

namespace table { namespace io {

class Persistable;
class PersistableFactory;

typedef std::vector<BaseCatalog> CatalogVector;


/**
 *  @brief A multi-catalog archive object used to load table::io::Persistable objects.
 *
 *  An InputArchive can be constructed directly from the catalogs produced by OutputArchive,
 *  or more usefully, read from a multi-extension FITS file.
 *
 *  @sa OutputArchive
 */
class InputArchive {
public:

    typedef std::map<int,PTR(Persistable)> Map;

    /**
     *  @brief Construct an archive from catalogs.
     */
    InputArchive(BaseCatalog const & index, CatalogVector const & dataCatalogs);

    /// Copy-constructor.  Does not deep-copy loaded Persistables.
    InputArchive(InputArchive const & other);

    /// Assignment.  Does not deep-copy loaded Persistables.
    InputArchive & operator=(InputArchive const & other);

    ~InputArchive();

    /**
     *  @brief Load the Persistable with the given ID and return it.
     *
     *  If the object has already been loaded once, the same instance will be returned again.
     */
    PTR(Persistable) get(int id) const;

    /// Load and return all objects in the archive.
    Map const & getAll() const;

    /**
     *  @brief Read an object from an already open FITS object.
     *
     *  @param[in]  fitsfile     FITS object to read from, already positioned at the desired HDU.
     */
    static InputArchive readFits(fits::Fits & fitsfile);

private:

    class Impl;

    InputArchive(PTR(Impl) impl);

    PTR(Impl) _impl;
};

}}}} // namespace lsst::afw::table::io

#endif // !AFW_TABLE_IO_InputArchive_h_INCLUDED
