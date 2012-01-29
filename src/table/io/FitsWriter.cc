// -*- lsst-c++ -*-

#include "boost/cstdint.hpp"

#include "lsst/afw/table/io/FitsWriter.h"
#include "lsst/afw/table/BaseTable.h"
#include "lsst/afw/table/BaseRecord.h"

namespace lsst { namespace afw { namespace table { namespace io {

namespace {

typedef FitsWriter::Fits Fits;

//----- Code to write FITS headers --------------------------------------------------------------------------

struct CountFlags {

    template <typename T>
    void operator()(SchemaItem<T> const &) const {}

    void operator()(SchemaItem<Flag> const &) const { ++n; }

    static int apply(Schema const & schema) {
        CountFlags counter = { 0 };
        schema.forEach(boost::ref(counter));
        return counter.n;
    }

    mutable int n;
};

struct ProcessSchema {

    template <typename T>
    void specialize(SchemaItem<T> const & item, int n) const {
        if (!item.field.getUnits().empty())
            fits->writeColumnKey("TUNIT", n, item.field.getUnits().c_str());
        fits->writeColumnKey("TCCLS", n, "Scalar", "Field template used by lsst.afw.table");
    }

    void specialize(SchemaItem<Angle> const & item, int n) const {
        if (!item.field.getUnits().empty())
            fits->writeColumnKey("TUNIT", n, item.field.getUnits().c_str());
        fits->writeColumnKey("TCCLS", n, "Angle", "Field template used by lsst.afw.table");
    }

    template <typename T>
    void specialize(SchemaItem< Array<T> > const & item, int n) const {
        if (!item.field.getUnits().empty())
            fits->writeColumnKey("TUNIT", n, item.field.getUnits().c_str());
        fits->writeColumnKey("TCCLS", n, "Array", "Field template used by lsst.afw.table");
    }

    void specialize(SchemaItem<Coord> const & item, int n) const {
        if (!item.field.getUnits().empty())
            fits->writeColumnKey("TUNIT", n, item.field.getUnits().c_str());
        fits->writeColumnKey("TCCLS", n, "Coord", "Field template used by lsst.afw.table");
    }

    template <typename T>
    void specialize(SchemaItem< Point<T> > const & item, int n) const {
        if (!item.field.getUnits().empty())
            fits->writeColumnKey("TUNIT", n, item.field.getUnits().c_str(), "{x, y}");
        fits->writeColumnKey("TCCLS", n, "Point", "Field template used by lsst.afw.table");
    }

    template <typename T>
    void specialize(SchemaItem< Moments<T> > const & item, int n) const {
        if (!item.field.getUnits().empty())
            fits->writeColumnKey("TUNIT", n, item.field.getUnits().c_str(), "{xx, yy, xy}");
        fits->writeColumnKey("TCCLS", n, "Moments", "Field template used by lsst.afw.table");
    }

    template <typename T>
    void specialize(SchemaItem< Covariance<T> > const & item, int n) const {
        if (!item.field.getUnits().empty())
            fits->writeColumnKey("TUNIT", n, item.field.getUnits().c_str(),
                                 "{(0,0), (0,1), (1,1), (0,2), (1,2), (2,2), ...}");
        fits->writeColumnKey("TCCLS", n, "Covariance", "Field template used by lsst.afw.table");
    }

    template <typename T>
    void specialize(SchemaItem< Covariance< Point<T> > > const & item, int n) const {
        if (!item.field.getUnits().empty())
            fits->writeColumnKey("TUNIT", n, item.field.getUnits().c_str(),
                                 "{(x,x), (x,y), (y,y)}");
        fits->writeColumnKey("TCCLS", n, "Covariance(Point)", "Field template used by lsst.afw.table");
    }

    template <typename T>
    void specialize(SchemaItem< Covariance< Moments<T> > > const & item, int n) const {
        if (!item.field.getUnits().empty())
            fits->writeColumnKey("TUNIT", n, item.field.getUnits().c_str(),
                                 "{(xx,xx), (xx,yy), (yy,yy), (xx,xy), (yy,xy), (xy,xy)}");
        fits->writeColumnKey("TCCLS", n, "Covariance(Moments)", "Field template used by lsst.afw.table");
    }

    template <typename T>
    void operator()(SchemaItem<T> const & item) const {
        std::string name = item.field.getName();
        std::replace(name.begin(), name.end(), '.', '_');
        int n = fits->addColumn<typename Field<T>::Element>(
            name.c_str(),
            item.field.getElementCount(),
            item.field.getDoc().c_str()
        );
        specialize(item, n);
    }

    void operator()(SchemaItem<Flag> const & item) const {
        std::string name = item.field.getName();
        std::replace(name.begin(), name.end(), '.', '_');
        fits->writeColumnKey("TFLAG", nFlags, name.c_str(), item.field.getDoc().c_str());
        ++nFlags;
    }

    static void apply(Fits & fits, Schema const & schema) {
        ProcessSchema f = { &fits, 0 };
        schema.forEach(boost::ref(f));
    }

    Fits * fits;
    mutable int nFlags;
};

//----- code for writing FITS records -----------------------------------------------------------------------

} // anonymous

struct FitsWriter::ProcessRecords {
    
    template <typename T>
    void operator()(SchemaItem<T> const & item) const {
        fits->writeTableArray(row, col, item.key.getElementCount(), record->getElement(item.key));
        ++col;
    }
    
    void operator()(SchemaItem<Flag> const & item) const {
        flags[bit] = record->get(item.key);
        ++bit;
    }

    ProcessRecords(Fits * fits_, Schema const & schema_, int nFlags_, std::size_t const & row_) :
        row(row_), col(0), bit(0), nFlags(nFlags_), fits(fits_), schema(schema_)
    {
        if (nFlags) flags.reset(new bool[nFlags]);
    }

    void apply(BaseRecord const * r) {
        record = r;
        col = 0;
        bit = 0;
        if (nFlags) ++col;
        schema.forEach(boost::ref(*this));
        if (nFlags) fits->writeTableArray(row, 0, nFlags, flags.get());
    }

    std::size_t const & row;
    mutable int col;
    mutable int bit;
    int nFlags;
    Fits * fits;
    boost::scoped_array<bool> flags;
    BaseRecord const * record;
    Schema schema;
};

void FitsWriter::_writeTable(CONST_PTR(BaseTable) const & table) {
    Schema schema = table->getSchema();
    _fits->createTable();
    _fits->checkStatus();
    int nFlags = CountFlags::apply(schema);
    if (nFlags > 0) {
        int n = _fits->addColumn<bool>("flags", nFlags, "bits for all Flag fields; see also TFLAGn");
        _fits->writeKey("FLAGCOL", n + 1, "Column number for the bitflags.");
    }
    ProcessSchema::apply(*_fits, schema);
    _row = -1;
    _processor = boost::make_shared<ProcessRecords>(_fits, schema, nFlags, _row);
}

void FitsWriter::_writeRecord(BaseRecord const & record) {
    ++_row;
    _processor->apply(&record);
}

}}}} // namespace lsst::afw::table::io