// -*- lsst-c++ -*-
#ifndef AFW_TABLE_IO_Persistable_h_INCLUDED
#define AFW_TABLE_IO_Persistable_h_INCLUDED

#include "boost/noncopyable.hpp"

#include "lsst/base.h"
#include "lsst/afw/table/Catalog.h"
#include "lsst/afw/table/io/OutputArchive.h"
#include "lsst/afw/table/io/InputArchive.h"

namespace lsst { namespace afw {

namespace fits {

class Fits;
class MemFileManager;

} // namespace fits

namespace table { namespace io {

/**
 *  @brief A base class for objects that can be persisted via afw::table::io Archive classes.
 *
 *  Inheriting from Persistable provides a public API for reading/writing individual objects to
 *  FITS that is fully defined in the base class, with derived classes only needing to implement
 *  the archive visitor interface.  It is expected that objects that contain multiple persistables
 *  (such as Exposures) will create their own InputArchives and OutputArchives, and use these
 *  to avoid writing the same object twice (which would otherwise be a big concern for future
 *  objects like ExposureCatalog and CoaddPsf).
 *
 *  Generally speaking, an abstract base class that inherits from Persistable should
 *  also inherit from PersistableFacade.
 *  A concrete class that inherits (indirectly) from Persistable should inherit (again) from
 *  PersistableFacade (though this just provides a slightly nicer interface to users), implement
 *  isPersistable and visitOutputArchive, and define a subclass of PersistenceFactory.
 *
 *  Persistable has no pure virtual member functions, and instead contains a default implementation
 *  that throws LogicErrorException when the user attempts to save them.
 */
class Persistable {
public:

    /**
     *  @brief Write the object to a regular FITS file.
     *
     *  @param[in] fileName     Name of the file to write to.
     *  @param[in] mode         If "w", any existing file with the given name will be overwritten.  If
     *                          "a", new HDUs will be appended to an existing file.
     */
    void writeFits(std::string const & fileName, std::string const & mode="w") const;

    /**
     *  @brief Write the object to a FITS image in memory.
     *
     *  @param[in] manager      Name of the file to write to.
     *  @param[in] mode         If "w", any existing file with the given name will be overwritten.  If
     *                          "a", new HDUs will be appended to an existing file.
     */
    void writeFits(fits::MemFileManager & manager, std::string const & mode="w") const;

#ifndef SWIG // only expose the higher-level interfaces to Python

    /**
     *  @brief Write the object to an already-open FITS object.
     *
     *  @param[in] fitsfile     Open FITS object to write to.
     */
    void writeFits(fits::Fits & fitsfile) const;

#endif // !SWIG

    /// @brief Return true if this particular object can be persisted using afw::table::io.
    virtual bool isPersistable() const { return false; }

    virtual ~Persistable() {}

protected:

    typedef io::OutputArchive OutputArchive; // convenient for derived classes not in afw::table::io

    /**
     *  @brief Return the unique name used to persist this object and look up its factory.
     *
     *  Must be less than ArchiveIndexSchema::MAX_NAME_LENGTHT characters.
     */
    virtual std::string getPersistenceName() const;

    /**
     *  @brief Write the object to one or more catalogs.
     *
     *  The handle object passed to this function provides an interface for adding new catalogs
     *  and adding nested objects to the same archive (while checking for duplicates).  See
     *  OutputArchive::Handle and OutputArchive::CatalogProxy for more information.
     */
    virtual void write(OutputArchive::Handle & handle) const;

    Persistable() {}

    Persistable(Persistable const & other) {}

    void operator=(Persistable const & other) {}

private:

    friend class io::OutputArchive;
    friend class io::InputArchive;

    template <typename T> friend class PersistableFacade;

    static PTR(Persistable) _readFits(std::string const & fileName, int hdu=0);

    static PTR(Persistable) _readFits(fits::MemFileManager & manager, int hdu=0);

    static PTR(Persistable) _readFits(fits::Fits & fitsfile);

};

/**
 *  @brief A CRTP facade class for subclasses of Persistable.
 *
 *  Derived classes should generally inherit from PersistableFacade at all levels,
 *  but only inherit from Persistable via the base class of each hierarchy.  For example,
 *  with Psfs:
 *  @code
 *  class Psf: public PersistableFacade<Psf>, public Persistable { ... };
 *  class DoubleGaussianPsf: public PersistableFacade<Psf>, public Psf { ... };
 *  @endcode
 *
 *  Inheriting from PersistableFacade is not required for any classes but the base of
 *  each hierarchy, but doing so can save users from having to do some dynamic_casts.
 *
 *  @note PersistableFacade should usually be the first class in a list of base classes;
 *  if it appears after a base class that inherits from different specialization of
 *  PersistableFacade, those base class member functions will hide the desired ones.
 */
template <typename T>
class PersistableFacade {
public:

#ifndef SWIG

    /**
     *  @brief Read an object from an already open FITS object.
     *
     *  @param[in]  fitsfile     FITS object to read from, already positioned at the desired HDU.
     */
    static PTR(T) readFits(fits::Fits & fitsfile) {
        return boost::dynamic_pointer_cast<T>(Persistable::_readFits(fitsfile));
    }

#endif // !SWIG

    /**
     *  @brief Read an object from a regular FITS file.
     *
     *  @param[in]  fileName     Name of the file to read.
     *  @param[in]  hdu          HDU to read, where 1 is the primary.  The special value of 0
     *                           skips the primary HDU if it is empty.
     */
    static PTR(T) readFits(std::string const & fileName, int hdu=0) {
        return boost::dynamic_pointer_cast<T>(Persistable::_readFits(fileName, hdu));
    }

    /**
     *  @brief Read an object from a FITS file in memory.
     *
     *  @param[in]  manager      Manager for the memory to read from.
     *  @param[in]  hdu          HDU to read, where 1 is the primary.  The special value of 0
     *                           skips the primary HDU if it is empty.
     */
    static PTR(T) readFits(fits::MemFileManager & manager, int hdu=0) {
        return boost::dynamic_pointer_cast<T>(Persistable::_readFits(manager, hdu));
    }

};

#ifndef SWIG

/**
 *  @brief A base class for factory classes used to reconstruct objects from records.
 *
 *  Classes that inherit from Persistable should also subclass PersistableFactory,
 *  and instantiate exactly one instance of the derived factory with static duration (usually
 *  the class and instance are both defined in an anonymous namespace in a source file).
 */
class PersistableFactory : private boost::noncopyable {
protected:
    typedef io::InputArchive InputArchive; // convenient for derived classes not in afw::table::io
    typedef io::CatalogVector CatalogVector;
public:

    /**
     *  @brief Constructor for the factory.
     *
     *  This should be called only once, and only on an object with static duration,
     *  as a pointer to the object will be put in a singleton registry.
     *
     *  The name must be globally unique with respect to *all* Persistables and be the
     *  same as Persistable::getPersistenceName().
     */
    explicit PersistableFactory(std::string const & name);

    /// @brief Construct a new object from the given InputArchive and vector of catalogs.
    virtual PTR(Persistable) read(InputArchive const & archive, CatalogVector const & catalogs) const = 0;

    /// @brief Return the factory that has been registered with the given name.
    static PersistableFactory const & lookup(std::string const & name);

    virtual ~PersistableFactory() {}
};

#endif // !SWIG

}}}} // namespace lsst::afw::table::io

#endif // !AFW_TABLE_IO_Persistable_h_INCLUDED