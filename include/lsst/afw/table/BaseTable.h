// -*- lsst-c++ -*-
#ifndef AFW_TABLE_BaseTable_h_INCLUDED
#define AFW_TABLE_BaseTable_h_INCLUDED

#include "boost/enable_shared_from_this.hpp"

#include "lsst/base.h"
#include "lsst/daf/base/Citizen.h"
#include "lsst/ndarray/Manager.h"
#include "lsst/afw/table/misc.h"
#include "lsst/afw/table/Schema.h"
#include "lsst/afw/table/io/FitsWriter.h"

namespace lsst { namespace afw { namespace table {

class BaseRecord;
class SchemaMapper;

class BaseTable 
#ifndef SWIG
: public boost::enable_shared_from_this<BaseTable>,
  private daf::base::Citizen
#endif
{
public:

    typedef BaseRecord Record;
    typedef VectorT<Record,BaseTable> Vector;
    typedef VectorT<Record const,BaseTable> ConstVector;

    /// @brief Number of records in each block when capacity is not given explicitly.
    static int nRecordsPerBlock;

    /**
     *  @brief Return a polymorphic deep copy.
     *
     *  Derived classes should reimplement by static-casting the output of _clone to a
     *  pointer-to-derived to simulate covariant return types.
     *
     *  Cloning a table does not clone its associated records; the table produced by clone()
     *  does not have any associated records.
     */
    PTR(BaseTable) clone() const { return _clone(); }

    /**
     *  @brief Default-construct an associated record.
     *
     *  Derived classes should reimplement by static-casting the output of _makeRecord to the
     *  appropriate BaseRecord subclass to simulate covariant return types.
     */
    PTR(BaseRecord) makeRecord() { return _makeRecord(); }

    /**
     *  @brief Deep copy a record, requiring that it have the same schema as this table.
     *
     *  Regardless of the type of the input record, the type of the output record will be the type
     *  associated with this table.
     *
     *  Derived classes should reimplement by static-casting the output of BaseTable::copyRecord to the
     *  appropriate BaseRecord subclass.
     *
     *  This is implemented using makeRecord and calling record.assign on the results; override those
     *  to change the behavior.
     */
    PTR(BaseRecord) copyRecord(BaseRecord const & input);

    /**
     *  @brief Deep copy a record, using a mapper to relate two schemas.
     *
     *  @copydetails BaseTable::copyRecord(BaseRecord const &)
     */
    PTR(BaseRecord) copyRecord(BaseRecord const & input, SchemaMapper const & mapper);
    
    /// @brief Return the table's schema.
    Schema getSchema() const { return _schema; }

    /**
     *  @brief Allocate contiguous space for new records in advance.
     *
     *  If a contiguous memory block for at least n additional records has already been allocated,
     *  this is a no-op.  If not, a new block will be allocated, and any remaining space on the old
     *  block will go to waste; this ensures the new records will be allocated contiguously.
     *
     *  Note that unlike std::vector::reserve, this does not factor in existing records in any way.
     */
    void preallocate(std::size_t n);

    /**
     *  @brief Construct a new table.
     *
     *  Because BaseTable is an abstract class, this actually returns a hidden trivial subclass
     *  (which is associated with a hidden trivial subclass of BaseRecord).
     *
     *  Hiding concrete table and record classes in anonymous namespaces is not required, but it
     *  makes it easier to ensure instances are always created within shared_ptrs,
     *  and it eliminates some multilateral friending that would otherwise be necessary.
     *  In some cases it may also serve as a form of pimpl, keeping class implementation details
     *  out of header files.
     */
    static PTR(BaseTable) make(Schema const & schema);

    virtual ~BaseTable() {}

protected:

    /// @brief Convenience functions for static-casting shared_from_this for use by derived classes.
    template <typename Derived>
    PTR(Derived) getSelf() {
        return boost::static_pointer_cast<Derived>(shared_from_this());
    }

    /// @brief Convenience functions for static-casting shared_from_this for use by derived classes.
    template <typename Derived>
    CONST_PTR(Derived) getSelf() const {
        return boost::static_pointer_cast<Derived const>(shared_from_this());
    }

    /// @brief Clone implementation with noncovariant return types.
    virtual PTR(BaseTable) _clone() const = 0;

    /// @brief Default-construct an associated record (protected implementation).
    virtual PTR(BaseRecord) _makeRecord() = 0;

    explicit BaseTable(Schema const & schema);

    BaseTable(BaseTable const & other) : daf::base::Citizen(other), _schema(other._schema) {}

private:
    
    friend class BaseRecord;
    friend class io::FitsWriter;

    // Called by BaseRecord ctor to fill in its _data and _manager members.
    void _initialize(BaseRecord & record);

    /*
     *  Called by BaseRecord dtor to notify the table when it is about to be destroyed.
     *
     *  This could allow the table to reclaim that space, but presently that requires
     *  more bookkeeping than it's worth unless this was the most recently allocated record.
     *  It does tell the table that it isn't contiguous anymore, preventing ColumnView access.
     */
    void _destroy(BaseRecord & record);

    // Tables may be copy-constructable (and are definitely cloneable), but are not assignable.
    void operator=(BaseTable const & other) { assert(false); }

    // Return a writer object that knows how to save in FITS format.
    virtual PTR(io::FitsWriter) makeFitsWriter(io::FitsWriter::Fits * fits) const;

    // All these are definitely private, not protected - we don't want derived classes mucking with them.
    Schema _schema;
    ndarray::Manager::Ptr _manager;
};

}}} // namespace lsst::afw::table

#endif // !AFW_TABLE_BaseTable_h_INCLUDED