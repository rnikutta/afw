// -*- lsst-c++ -*-
#ifndef AFW_TABLE_TableBase_h_INCLUDED
#define AFW_TABLE_TableBase_h_INCLUDED

#include "lsst/base.h"
#include "lsst/afw/table/Schema.h"
#include "lsst/afw/table/ColumnView.h"
#include "lsst/afw/table/RecordBase.h"
#include "lsst/afw/table/IteratorBase.h"
#include "lsst/afw/table/IdFactory.h"

namespace lsst { namespace afw { namespace table {

class SchemaMapper;

/**
 *  @brief Base class containing most of the implementation for tables.
 *
 *  Final table classes should generally not inherit from TableBase directly,
 *  and instead should inherit from TableInterface.
 *
 *  Most of the implementation of derived table classes is provided here
 *  in the form of protected member functions that will need to be wrapped
 *  into public member functions by derived classes.
 *
 *  Data is shared between records and tables, but the assertion-based
 *  modification flags are not shared.
 */
class TableBase : protected ModificationFlags {
public:

    /// @brief Number of records in each block when capacity is not given explicitly.
    static int nRecordsPerBlock;

    /// @brief Return the schema for the table's fields.  
    Schema getSchema() const;

    /// @brief Return true if all records are allocated in a single contiguous blocks.
    bool isConsolidated() const;

    /**
     *  @brief Consolidate the table in-place into a single contiguous block.
     *
     *  This does not invalidate any existing records or iterators, but existing
     *  records and iterators will no longer be associated with this table.
     *
     *  This will also reallocate the table even if the table is already consolidated.
     *  
     *  @param[in] extraCapacity  Number of additional records to allocate space for
     *                            as part of the same block.  Adding N additional records
     *                            where N is <= extraCapacity will not cause the table
     *                            to become unconsolidated.
     */
    void consolidate(int extraCapacity=0);

    /**
     *  @brief Return a strided-array view into the columns of the table.
     *
     *  This will raise LogicErrorException if the table is not consolidated.
     */
    ColumnView getColumnView() const;

    /// @brief Return the number of records in the table.
    int getRecordCount() const;

    /// @brief Disable modifications of the sort defined by the given bit.
    void disable(ModificationFlags::Bit n) { unsetBit(n); }

    /// @brief Disable all modifications.
    void makeReadOnly() { unsetAll(); }

    /// Destructor is explicit because class holds a shared_ptr to an incomplete class.
    ~TableBase();

protected:

    /**
     *  @brief Standard constructor for TableBase.
     *
     *  @param[in] schema            Schema that defines the fields, offsets, and record size for the table.
     *  @param[in] capacity          Number of records to pre-allocate space for the first block.
     *  @param[in] idFactory         Factory class to generate record IDs when they are not explicitly given.
     *                               If empty, defaults to a simple counter that starts at 1.
     *  @param[in] aux               A pointer containing extra arbitrary data for the table.
     *  @param[in] flags             Bitflags for assertion-based modification protection (see the
     *                               ModificationFlags class for more information).
     */
    TableBase(
        Schema const & schema,
        int capacity,
        PTR(IdFactory) const & idFactory = PTR(IdFactory)(),
        PTR(AuxBase) const & aux = PTR(AuxBase)(),
        ModificationFlags const & flags = ModificationFlags::all()
    );

    /**
     *  @brief Shared copy constructor.
     *
     *  All aspects of the table except the modification flags are shared between the two tables.
     *  The modification flags will be copied as well, but can then be changed separately.
     */
    TableBase(TableBase const & other) : ModificationFlags(other), _impl(other._impl) {}

    /**
     *  @brief Insert an existing record into the table.
     *
     *  The optional initial "hint" argument is analogous to the hint argument in std::set::insert
     *  or std::map::insert; a good hint makes insertion a constant-time operation rather than
     *  logarithmic.
     *
     *  The Schema of the given record must be equal to the Schema of the table.
     *
     *  The record will be deep-copied, aside from auxiliary data (which will be shallow-copied).
     *  If non-unique IDs are encountered, the existing record with that ID will be overwritten.
     *
     *  @return an iterator to the record just added (or overwritten).
     */
    IteratorBase _insert(IteratorBase const & hint, RecordBase const & record) const;

    /**
     *  @brief Insert an existing record into the table, using a SchemaMapper to only copy certain fields.
     *
     *  The optional initial "hint" argument is analogous to the hint argument in std::set::insert
     *  or std::map::insert; a good hint makes insertion a constant-time operation rather than
     *  logarithmic.
     *
     *  The mapper's input Schema must match that of the given record, while the output Schema must match
     *  the input record.
     *
     *  The record will be deep-copied, aside from auxiliary data (which will be shallow-copied).
     *  If non-unique IDs are encountered, the existing record with that ID will be overwritten.
     *
     *  @return an iterator to the record just added (or overwritten).
     */
    IteratorBase _insert(
        IteratorBase const & hint, RecordBase const & record, SchemaMapper const & mapper
    ) const;

    //@{
    /**
     *  @brief Remove the record pointed at by the given iterator.
     *
     *  After unlinking, the passed iterator can still be dereferenced and the record will remain valid,
     *  but the result of incrementing the iterator is undefined.
     */
    IteratorBase _unlink(IteratorBase const & iter) const;
    //@}

    //@{
    /// @brief Return begin and end iterators that go through the table in ID order.
    IteratorBase _begin() const;
    IteratorBase _end() const;
    //@}

    /// @brief Return the record with the given ID or throw NotFoundException.
    RecordBase _get(RecordId id) const;

    /// @brief Return an iterator to the record with the given ID or throw NotFoundException.
    IteratorBase _find(RecordId id) const;

    /// @brief Create and add a new record with an ID generated by the table's IdFactory.
    RecordBase _addRecord(PTR(AuxBase) const & aux = PTR(AuxBase)()) const;

    /// @brief Create and add a new record with an explicit RecordId.
    RecordBase _addRecord(RecordId id, PTR(AuxBase) const & aux = PTR(AuxBase)()) const;

    /// @brief Return the table's auxiliary data.
    PTR(AuxBase) getAux() const;

private:
    PTR(detail::TableImpl) _impl;
};

}}} // namespace lsst::afw::table

#endif // !AFW_TABLE_TableBase_h_INCLUDED
