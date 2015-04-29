// -*- lsst-c++ -*-
#ifndef AFW_TABLE_IO_FitsReader_h_INCLUDED
#define AFW_TABLE_IO_FitsReader_h_INCLUDED

#include "lsst/afw/fits.h"
#include "lsst/afw/table/Schema.h"
#include "lsst/afw/table/io/InputArchive.h"

namespace lsst { namespace afw { namespace table { namespace io {

/// Helper class used by FitsReader; not private so classes in anonymous namespaces can see it.
class FieldReader;

/**
 *  @brief A Reader subclass for FITS binary tables.
 *
 *  FitsReader itself provides the implementation for reading standard FITS binary tables
 *  (with a limited subset of FITS column types), but it also allows subclasses to be used
 *  instead, depending on what's actually in the FITS file.  If the FITS header has the key
 *  "AFW_TABLE" with a value other than "BASE", FitsReader::make consults a registry of
 *  and constructs the subclass corresponding to that key.  This means the type of
 *  records/tables loaded correctly depends on the file itself, rather than the caller.
 *  For instance, if you load a FITS table corresponding to a saved SourceCatalog using
 *  BaseCatalog::readFits, you'll actually get a BaseCatalog whose record are actually
 *  SourceRecords and whose table is actually a SourceTable.  On the other hand, if you
 *  try to load a non-Source FITS table into a SourceCatalog, you'll get an exception
 *  when it tries to dynamic_cast the table to a SourceTable.
 */
class FitsReader {
public:

    typedef afw::fits::Fits Fits;

    /**
     *  @brief Factory class used to construct FitsReaders.
     *
     *  The constructor for Factory puts a raw pointer to itself in a global registry.
     *  This means Factory and its subclasses should only be constructed as namespace-scope
     *  objects (so they never go out of scope, and automatically get registered).
     *
     *  Subclasses should use this via its derived template class FactoryT.
     */
    class Factory {
    public:

        /// Create a new FITS reader from a cfitsio pointer holder and (optional) input archive and flags
        virtual PTR(FitsReader) operator()(Fits * fits, PTR(InputArchive) archive, int flags) const = 0;

        virtual ~Factory() {}

        /// Create a factory that will be used when the AFW_TYPE fits key matches the given name.
        explicit Factory(std::string const & name);

    };

    /**
     *  @brief Subclass for Factory that constructs a FitsReader.
     *
     *  Subclasses should use this by providing a the appropriate constructor and then declaring
     *  a static data member or namespace-scope FactoryT instance templated over the subclass type.
     *  This will register the subclass so it can be used with FitsReader::make.
     */
    template <typename ReaderT>
    class FactoryT : public Factory {
    public:

        /// Create a new FITS reader from a cfitsio pointer holder and (optional) input archive and flags
        virtual PTR(FitsReader) operator()(Fits * fits, PTR(InputArchive) archive, int flags) const {
            return boost::make_shared<ReaderT>(fits, archive, flags);
        }

        /// Create a factory that will be used when the AFW_TYPE fits key matches the given name.
        explicit FactoryT(std::string const & name) : Factory(name) {}

    };

    /**
     *  @brief Look for the header key (AFW_TYPE) that tells us the type of the FitsReader to use,
     *         then make it using the registered factory.
     */
    static PTR(FitsReader) make(Fits * fits, PTR(io::InputArchive) archive, int flags);

    /**
     *  @brief Entry point for reading FITS files into arbitrary containers.
     *
     *  This does the work of opening the file, calling FitsReader::make, and then calling
     *  FitsReader::_read.
     */
    template <typename ContainerT, typename SourceT>
    static ContainerT apply(SourceT & source, int hdu, int flags) {
        Fits fits(source, "r", Fits::AUTO_CLOSE | Fits::AUTO_CHECK);
        fits.setHdu(hdu);
        return apply<ContainerT>(fits, flags);
    }

    /// @brief Low-level entry point for reading FITS files into arbitrary containers.
    template <typename ContainerT>
    static ContainerT apply(Fits & fits, PTR(io::InputArchive) archive=PTR(io::InputArchive)(), int flags=0) {
        PTR(FitsReader) reader = make(&fits, archive, flags);
        return reader->template _read<ContainerT>();
    }

    /// @brief Low-level entry point for reading FITS files into arbitrary containers.
    template <typename ContainerT>
    static ContainerT apply(Fits & fits, int flags=0) {
        PTR(FitsReader) reader = make(&fits, PTR(io::InputArchive)(), flags);
        return reader->template _read<ContainerT>();
    }

    /**
     *  @brief Construct from a wrapped cfitsio pointer and (ignored) InputArchive.
     *
     *  Subclasses that require an InputArchive should accept the one that is passed in,
     *  but may need to construct their own from the HDUs following the catalog HDU(s)
     *  if this pointer is null.
     */
    explicit FitsReader(Fits * fits, PTR(InputArchive), int flags) : _fits(fits), _flags(flags) {}

    ~FitsReader(); // needs to go in .cc so it can see FieldReader dtor

private:

    /**
     *  @brief Load an on-disk table into a container.
     *
     *  The container must be a specialized table container (like CatalogT):
     *   - It must be constructable from a single table shared_ptr argument.
     *   - It must have an insert member function that takes an position
     *     iterator and a record shared_ptr.
     */
    template <typename ContainerT>
    ContainerT _read() {
    #if 1
        // Work around a clang++ version 3.0 (tags/Apple/clang-211.12) bug with shared_ptr reference counts
        PTR(typename ContainerT::Table) table;
        {
            PTR(BaseTable) tmpTable = _readTable();
            table = boost::dynamic_pointer_cast<typename ContainerT::Table>(tmpTable);
        }
    #else
        PTR(typename ContainerT::Table) table
            = boost::dynamic_pointer_cast<typename ContainerT::Table>(_readTable());
    #endif
        if (!table) {
            throw LSST_EXCEPT(
                lsst::pex::exceptions::RuntimeError,
                "Container's table type is not compatible with on-disk table type."
            );
        }
        ContainerT container(table);
        PTR(BaseRecord) record = _readRecord(table);
        while (record) {
            container.insert(
                container.end(),
                boost::static_pointer_cast<typename ContainerT::Record>(record)
            );
            record = _readRecord(table);
        }
        return container;
    }


protected:

    /**
     *  @brief Create a new table of the appropriate type.
     *
     *  The result may be an instance of a subclass of BaseTable.
     */
    virtual PTR(BaseTable) _readTable();

    /**
     *  @brief Read an individual record, creating it with the given table.
     *
     *  The result may be an instance of a subclass of BaseRecord.  The table will have just been loaded
     *  with _readSchema; these are separated in order to allow subclasses to delegate to base
     *  class implementations more effectively.
     */
    virtual PTR(BaseRecord) _readRecord(PTR(BaseTable) const & table);

    /// @brief Should be called by any reimplementation of _readTable.
    void _startRecords(BaseTable & table);

    struct ProcessRecords;

    Fits * _fits;         // cfitsio pointer in a conveniencer wrapper
    int _flags;           // subclass-defined flags to control FITS reading
    std::size_t _row;     // which row we're currently reading

    typedef std::vector<PTR(FieldReader)> FieldReaderVector;

    // Implementation for Schema's constructors that take PropertyLists;
    // it's here to keep FITS-related code a little more centralized.
    static FieldReaderVector _readSchema(
        Schema & schema,
        daf::base::PropertyList & metadata,
        bool stripMetadata
    );

    friend class afw::table::Schema;

    FieldReaderVector _fields;

    std::size_t _nRows;   // how many total records there are in the FITS table
};

}}}} // namespace lsst::afw::table::io

#endif // !AFW_TABLE_IO_FitsReader_h_INCLUDED
