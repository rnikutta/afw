// -*- lsst-c++ -*-

#include <cstdio>

#include "boost/regex.hpp"
#include "boost/lexical_cast.hpp"
#include "boost/cstdint.hpp"
#include "boost/multi_index_container.hpp"
#include "boost/multi_index/sequenced_index.hpp"
#include "boost/multi_index/ordered_index.hpp"
#include "boost/multi_index/member.hpp"
#include "boost/math/special_functions/round.hpp"

#include "lsst/afw/table/io/FitsReader.h"
#include "lsst/afw/table/BaseRecord.h"
#include "lsst/afw/table/BaseTable.h"

namespace lsst { namespace afw { namespace table { namespace io {

/*
 *  This file contains most of the code for reading FITS binary tables.  There's also a little
 *  in Source.cc, where we read the stuff that's specific to SourceTable/SourceRecord (footprints
 *  and aliases).
 */

class FieldReader {
public:

    FieldReader() {}

    // Neither copyable nor moveable.
    FieldReader(FieldReader const &) = delete;
    FieldReader(FieldReader &&) = delete;
    FieldReader & operator=(FieldReader const &) = delete;
    FieldReader & operator=(FieldReader &&) = delete;

    virtual void readCell(
        std::size_t row, BaseRecord & record,
        fits::Fits & fits, boost::scoped_array<bool> const & flagData
    ) const = 0;

    virtual ~FieldReader() {}

};

namespace {

typedef FitsReader::Fits Fits;

template <typename T>
class StandardReader : public FieldReader {
public:

    StandardReader(
        std::size_t col, Schema & schema,
        std::string const & name, std::string const & doc, std::string const & unit,
        FieldBase<T> const & base = FieldBase<T>()
    ) :
        _col(col), _key(schema.addField<T>(name, doc, unit, base))
    {}

    virtual void readCell(
        std::size_t row, BaseRecord & record,
        Fits & fits, boost::scoped_array<bool> const & flagData
    ) const {
        fits.readTableArray(row, _col, _key.getElementCount(), record.getElement(_key));
    }

private:
    std::size_t _col;
    Key<T> _key;
};

class StringReader : public FieldReader {
public:

    StringReader(
        std::size_t col, Schema & schema,
        std::string const & name, std::string const & doc, std::string const & unit,
        int size
    ) :
        _col(col), _key(schema.addField<std::string>(name, doc, unit, size))
    {}

    virtual void readCell(
        std::size_t row, BaseRecord & record,
        Fits & fits, boost::scoped_array<bool> const & flagData
    ) const {
        std::string s;
        fits.readTableScalar(row, _col, s);
        record.set(_key, s);
    }

private:
    std::size_t _col;
    Key<std::string> _key;
};

class FlagReader : public FieldReader {
public:

    FlagReader(
        std::size_t  bit, Schema & schema,
        std::string const & name, std::string const & doc
    ) :
        _bit(bit), _key(schema.addField<Flag>(name, doc))
    {}

    virtual void readCell(
        std::size_t row, BaseRecord & record,
        Fits & fits, boost::scoped_array<bool> const & flagData
    ) const {
        record.set(_key, flagData[_bit]);
    }

private:
    std::size_t _bit;
    Key<Flag> _key;
};

template <typename T>
class VariableLengthArrayReader : public FieldReader {
public:

    VariableLengthArrayReader(
        std::size_t col, Schema & schema,
        std::string const & name, std::string const & doc, std::string const & unit
    ) :
        _col(col), _key(schema.addField<Array<T>>(name, doc, unit, 0))
    {}

    virtual void readCell(
        std::size_t row, BaseRecord & record,
        Fits & fits, boost::scoped_array<bool> const & flagData
    ) const {
        int size = fits.getTableArraySize(row, _col);
        ndarray::Array<T,1,1> array = ndarray::allocate(size);
        fits.readTableArray(row, _col, size, array.getData());
        record.set(_key, array);
    }

private:
    std::size_t _col;
    Key<Array<T>> _key;
};

// ------------ FITS header to Schema implementation -------------------------------------------------------

/*
 *  We read FITS headers in two stages - first we read all the information we care about into
 *  a temporary structure (FitsSchema) we can access more easily than a raw FITS header,
 *  and then we iterate through that to fill the actual Schema object.
 *
 *  The driver code is at the bottom of this section; it's easier to understand if you start there
 *  and work your way up.
 */

// A structure that describes a field as a bunch of strings read from the FITS header.
struct FitsSchemaItem {
    int col;             // column number (0-indexed)
    int bit;             // bit number for flag fields; -1 for others.
    std::string name;    // name of the field (from TTYPE keys)
    std::string units;   // field units (from TUNIT keys)
    std::string doc;     // field docs (from comments on TTYPE keys)
    std::string format;  // FITS column format code (from TFORM keys)
    std::string cls;     // which field class to use (from our own TCCLS keys)

    // Add the field defined by the strings to a schema.
    PTR(FieldReader) addField(Schema & schema) const {
        static boost::regex const regex("(\\d+)?([PQ])?(\\u)\\(?(\\d)*\\)?", boost::regex::perl);
        // start by parsing the format; this tells the element type of the field and the number of elements
        boost::smatch m;
        if (!boost::regex_match(format, m, regex)) {
            throw LSST_EXCEPT(
                afw::fits::FitsError,
                (boost::format("Could not parse TFORM value for field '%s': '%s'.") % name % format).str()
            );
        }
        int size = 1;
        if (m[1].matched) {
            size = boost::lexical_cast<int>(m[1].str());
        }
        char code = m[3].str()[0];
        if (m[2].matched) {
            // P or Q presence indicates a variable-length array, which we can get by just setting the
            // size to zero and letting the rest of the logic run its course.
            size = 0;
        }
        // switch code over FITS codes that correspond to different element types
        switch (code) {
        case 'I': // 16-bit integers - can only be scalars or Arrays (we assume they're unsigned, since
                  // that's all we ever write, and CFITSIO will complain later if they aren't)
            if (size == 1) {
                if (cls == "Array") {
                    return boost::make_shared<StandardReader<Array<boost::uint16_t>>>(
                        col, schema, name, doc, units, size
                    );
                }
                return boost::make_shared<StandardReader<boost::uint16_t>>(col, schema, name, doc, units);
            }
            if (size == 0) {
                return boost::make_shared<VariableLengthArrayReader<boost::uint16_t>>(
                    col, schema, name, doc, units
                );
            }
            return boost::make_shared<StandardReader<Array<boost::uint16_t>>>(
                col, schema, name, doc, units, size
            );
        case 'J': // 32-bit integers - can only be scalars, Point fields, or Arrays
            if (size == 0) {
                return boost::make_shared<VariableLengthArrayReader<boost::int32_t>>(
                    col, schema, name, doc, units
                );
            }
            if (cls == "Point") {
                return boost::make_shared<StandardReader<Point<boost::int32_t>>>(
                    col, schema, name, doc, units
                );
            }
            if (size > 1 || cls == "Array") {
                return boost::make_shared<StandardReader<Array<boost::int32_t>>>(
                    col, schema, name, doc, units, size
                );
            }
            return boost::make_shared<StandardReader<boost::int32_t>>(
                col, schema, name, doc, units
            );
        case 'K': // 64-bit integers - can only be scalars.
            if (size == 1) {
                return boost::make_shared<StandardReader<boost::int64_t>>(
                    col, schema, name, doc, units
                );
            }
        case 'E': // floats and doubles can be any number of things; delegate to a separate function
            return addFloatField<float>(schema, size);
        case 'D':
            return addFloatField<double>(schema, size);
        case 'A': // strings
            return boost::make_shared<StringReader>(col, schema, name, doc, units, size);
        default:
            // We throw if we encounter a column type we can't handle.
            // This raises probem when we want to save footprints as variable length arrays
            // later, so we add the nCols argument to Reader::_readSchema to allow SourceFitsReader
            // to call FitsReader::_readSchema in a way that prevents it from ever getting here.
            throw LSST_EXCEPT(
                afw::fits::FitsError,
                (boost::format("Unsupported FITS column type: '%s'.") % format).str()
            );
        }
    }

    // Add a field with a float or double element type to the schema.
    template <typename U>
    PTR(FieldReader) addFloatField(Schema & schema, int size) const {
        if (size == 0) {
            return boost::make_shared<VariableLengthArrayReader<U>>(
                col, schema, name, doc, units
            );
        }
        if (size == 1) {
            if (cls == "Angle") {
                return boost::make_shared<StandardReader<Angle>>(col, schema, name, doc, units);
            }
            if (cls == "Array") {
                return boost::make_shared<StandardReader<Array<U>>>(col, schema, name, doc, units, 1);
            }
            if (cls == "Covariance") {
                return boost::make_shared<StandardReader<Covariance<float>>>(
                    col, schema, name, doc, units, 1
                );
            }
            return boost::make_shared<StandardReader<U>>(col, schema, name, doc, units);
        } else if (size == 2) {
            if (cls == "Point") {
                return boost::make_shared<StandardReader<Point<double>>>(col, schema, name, doc, units);
            }
            if (cls == "Coord") {
                return boost::make_shared<StandardReader<Coord>>(col, schema, name, doc, units);
            }
        } else if (size == 3) {
            if (cls == "Moments") {
                return boost::make_shared<StandardReader<Moments<double>>>(col, schema, name, doc, units);
            }
            if (cls == "Covariance(Point)") {
                return boost::make_shared<StandardReader<Covariance<Point<float>>>>(
                    col, schema, name, doc, units
                );
            }
        } else if (size == 6) {
            if (cls == "Covariance(Moments)") {
                return boost::make_shared<StandardReader<Covariance<Moments<float>>>>(
                    col, schema, name, doc, units
                );
            }
        }
        if (cls == "Covariance") {
            double v = 0.5 * (std::sqrt(1 + 8 * size) - 1);
            int n = boost::math::iround(v);
            if (n * (n + 1) != size * 2) {
                throw LSST_EXCEPT(
                    afw::fits::FitsError,
                    "Covariance field has invalid size."
                );
            }
            return boost::make_shared<StandardReader<Covariance<float>>>(
                col, schema, name, doc, units, size
            );
        }
        return boost::make_shared<StandardReader<Array<U>>>(col, schema, name, doc, units, size);
    }


    FitsSchemaItem(int col_, int bit_) : col(col_), bit(bit_) {}
};

// A quirk of Boost.MultiIndex (which we use for our container of FitsSchemaItems)
// that you have to use a special functor (like this one) to set data members
// in a container with set indices (because setting those values might require 
// the element to be moved to a different place in the set).  Check out
// the Boost.MultiIndex docs for more information.
template <std::string FitsSchemaItem::*Member>
struct SetFitsSchemaString {
    void operator()(FitsSchemaItem & item) {
        item.*Member = _v;
    }
    explicit SetFitsSchemaString(std::string const & v) : _v(v) {}
private:
    std::string const & _v;
};

// A container class (based on Boost.MultiIndex) that provides two sort orders,
// on column number and on flag bit.  This allows us to insert fields into the
// schema in the correct order, regardless of which order they appear in the
// FITS header.
struct FitsSchema {
    typedef boost::multi_index_container<
        FitsSchemaItem,
        boost::multi_index::indexed_by<
            boost::multi_index::ordered_non_unique<
                boost::multi_index::member<FitsSchemaItem,int,&FitsSchemaItem::col>
                >,
            boost::multi_index::ordered_non_unique<
                boost::multi_index::member<FitsSchemaItem,int,&FitsSchemaItem::bit>
                >,
            boost::multi_index::sequenced<>
            >
        > Container;

    // Typedefs for the special functors used to set data members.
    typedef SetFitsSchemaString<&FitsSchemaItem::name> SetName;
    typedef SetFitsSchemaString<&FitsSchemaItem::units> SetUnits;
    typedef SetFitsSchemaString<&FitsSchemaItem::doc> SetDoc;
    typedef SetFitsSchemaString<&FitsSchemaItem::format> SetFormat;
    typedef SetFitsSchemaString<&FitsSchemaItem::cls> SetCls;

    // Typedefs for the different indices.
    typedef Container::nth_index<0>::type ColSet;
    typedef Container::nth_index<1>::type BitSet;
    typedef Container::nth_index<2>::type List;

    // Getters for the different indices.
    ColSet & asColSet() { return container.get<0>(); }
    BitSet & asBitSet() { return container.get<1>(); }
    List & asList() { return container.get<2>(); }

    Container container;
};

} // anonymous

FitsReader::FieldReaderVector FitsReader::_readSchema(
    Schema & schema,
    daf::base::PropertyList & metadata,
    bool stripMetadata
) {

    FitsSchema intermediate;
    // Set the table version.  If AFW_TABLE_VERSION tag exists, use that
    // If not, set to 0 if it has an AFW_TYPE, Schema default otherwise (DM-590)
    int version = 0;
    if (!metadata.exists("AFW_TYPE")) {
        version = lsst::afw::table::Schema::VERSION;
    }
    version = metadata.get("AFW_TABLE_VERSION", version);
    if (stripMetadata && metadata.exists("AFW_TABLE_VERSION")) metadata.remove("AFW_TABLE_VERSION");
    int flagCol = metadata.get("FLAGCOL", 0);
    if (flagCol > 0) {
        metadata.remove("FLAGCOL");
        metadata.remove((boost::format("TTYPE%d") % flagCol).str());
        metadata.remove((boost::format("TFORM%d") % flagCol).str());
    }
    --flagCol; // switch from 1-indexed to 0-indexed

    // read aliases stored in the new, expected way
    try {
        std::vector<std::string> rawAliases = metadata.getArray<std::string>("ALIAS");
        for (std::vector<std::string>::const_iterator i = rawAliases.begin(); i != rawAliases.end(); ++i) {
            std::size_t pos = i->find_first_of(':');
            if (pos == std::string::npos) {
                throw LSST_EXCEPT(
                    afw::fits::FitsError,
                    (boost::format("Malformed alias definition: '%s'") % (*i)).str()
                );
            }
            schema.getAliasMap()->set(i->substr(0, pos), i->substr(pos+1, std::string::npos));
        }
    } catch (pex::exceptions::NotFoundError &) {
        // if there are no aliases, just move on
    }
    metadata.remove("ALIAS");

    if (version == 0) {
        // Read slots saved using an old mechanism in as aliases, since the new slot mechanism delegates
        // slot definition to the AliasMap.
        static boost::array<std::pair<std::string,std::string>,6> oldSlotKeys = {
            {
                std::make_pair("PSF_FLUX", "slot_PsfFlux"),
                std::make_pair("AP_FLUX", "slot_ApFlux"),
                std::make_pair("INST_FLUX", "slot_InstFlux"),
                std::make_pair("MODEL_FLUX", "slot_ModelFlux"),
                std::make_pair("CENTROID", "slot_Centroid"),
                std::make_pair("SHAPE", "slot_Shape")
            }
        };
        for (std::size_t i = 0; i < oldSlotKeys.size(); ++i) {
            std::string target = metadata.get(oldSlotKeys[i].first + "_SLOT", std::string(""));
            std::replace(target.begin(), target.end(), '_', '.');
            if (!target.empty()) {
                schema.getAliasMap()->set(oldSlotKeys[i].second, target);
                if (stripMetadata) {
                    metadata.remove(oldSlotKeys[i].first);
                    metadata.remove(oldSlotKeys[i].first + "_ERR_SLOT");
                    metadata.remove(oldSlotKeys[i].first + "_FLAG_SLOT");
                }
            }
        }
    }

    std::vector<std::string> keyList = metadata.getOrderedNames();
    for (std::vector<std::string>::const_iterator key = keyList.begin(); key != keyList.end(); ++key) {
        if (key->compare(0, 5, "TTYPE") == 0) {
            int col = boost::lexical_cast<int>(key->substr(5)) - 1;
            FitsSchema::ColSet::iterator i = intermediate.asColSet().lower_bound(col);
            if (i == intermediate.asColSet().end() || i->col != col) {
                i = intermediate.asColSet().insert(i, FitsSchemaItem(col, -1));
            }
            std::string v = metadata.get<std::string>(*key);
            intermediate.asColSet().modify(i, FitsSchema::SetName(v));
            if (i->doc.empty()) // don't overwrite if already set with TDOCn
                intermediate.asColSet().modify(i, FitsSchema::SetDoc(metadata.getComment(*key)));
            if (stripMetadata) metadata.remove(*key);
        } else if (key->compare(0, 5, "TFLAG") == 0) {
            int bit = boost::lexical_cast<int>(key->substr(5)) - 1;
            FitsSchema::BitSet::iterator i = intermediate.asBitSet().lower_bound(bit);
            if (i == intermediate.asBitSet().end() || i->bit != bit) {
                i = intermediate.asBitSet().insert(i, FitsSchemaItem(-1, bit));
            }
            std::string v = metadata.get<std::string>(*key);
            intermediate.asBitSet().modify(i, FitsSchema::SetName(v));
            if (i->doc.empty()) // don't overwrite if already set with TFDOCn
                intermediate.asBitSet().modify(i, FitsSchema::SetDoc(metadata.getComment(*key)));
            if (stripMetadata) metadata.remove(*key);
        } else if (key->compare(0, 4, "TDOC") == 0) {
            int col = boost::lexical_cast<int>(key->substr(4)) - 1;
            FitsSchema::ColSet::iterator i = intermediate.asColSet().lower_bound(col);
            if (i == intermediate.asColSet().end() || i->col != col) {
                i = intermediate.asColSet().insert(i, FitsSchemaItem(col, -1));
            }
            intermediate.asColSet().modify(i, FitsSchema::SetDoc(metadata.get<std::string>(*key)));
            if (stripMetadata) metadata.remove(*key);
        } else if (key->compare(0, 5, "TFDOC") == 0) {
            int bit = boost::lexical_cast<int>(key->substr(5)) - 1;
            FitsSchema::BitSet::iterator i = intermediate.asBitSet().lower_bound(bit);
            if (i == intermediate.asBitSet().end() || i->bit != bit) {
                i = intermediate.asBitSet().insert(i, FitsSchemaItem(-1, bit));
            }
            intermediate.asBitSet().modify(i, FitsSchema::SetDoc(metadata.get<std::string>(*key)));
            if (stripMetadata) metadata.remove(*key);
        } else if (key->compare(0, 5, "TUNIT") == 0) {
            int col = boost::lexical_cast<int>(key->substr(5)) - 1;
            FitsSchema::ColSet::iterator i = intermediate.asColSet().lower_bound(col);
            if (i == intermediate.asColSet().end() || i->col != col) {
                i = intermediate.asColSet().insert(i, FitsSchemaItem(col, -1));
            }
            intermediate.asColSet().modify(i, FitsSchema::SetUnits(metadata.get<std::string>(*key)));
            if (stripMetadata) metadata.remove(*key);
        } else if (key->compare(0, 5, "TCCLS") == 0) {
            int col = boost::lexical_cast<int>(key->substr(5)) - 1;
            FitsSchema::ColSet::iterator i = intermediate.asColSet().lower_bound(col);
            if (i == intermediate.asColSet().end() || i->col != col) {
                i = intermediate.asColSet().insert(i, FitsSchemaItem(col, -1));
            }
            intermediate.asColSet().modify(i, FitsSchema::SetCls(metadata.get<std::string>(*key)));
            if (stripMetadata) metadata.remove(*key);
        } else if (key->compare(0, 5, "TFORM") == 0) {
            int col = boost::lexical_cast<int>(key->substr(5)) - 1;
            FitsSchema::ColSet::iterator i = intermediate.asColSet().lower_bound(col);
            if (i == intermediate.asColSet().end() || i->col != col) {
                i = intermediate.asColSet().insert(i, FitsSchemaItem(col, -1));
            }
            intermediate.asColSet().modify(i, FitsSchema::SetFormat(metadata.get<std::string>(*key)));
            if (stripMetadata) metadata.remove(*key);
        } else if (key->compare(0, 5, "TZERO") == 0) {
            if (stripMetadata) metadata.remove(*key);
        } else if (key->compare(0, 5, "TSCAL") == 0) {
            if (stripMetadata) metadata.remove(*key);
        } else if (key->compare(0, 5, "TNULL") == 0) {
            if (stripMetadata) metadata.remove(*key);
        } else if (key->compare(0, 5, "TDISP") == 0) {
            if (stripMetadata) metadata.remove(*key);
        }
    }


    FitsReader::FieldReaderVector fields;
    fields.reserve(intermediate.asList().size());
    for (auto i = intermediate.asList().begin(); i != intermediate.asList().end(); ++i) {
        if (i->bit >= 0) {
            fields.push_back(boost::make_shared<FlagReader>(i->bit, schema, i->name, i->doc));
        } else if (i->col != flagCol) {
            fields.push_back(i->addField(schema));
        }
    }
    return fields;
}

void FitsReader::_startRecords(BaseTable & table) {
    PTR(daf::base::PropertyList) metadata = table.getMetadata();
    if (metadata) {
        if (metadata->exists("AFW_TYPE")) metadata->remove("AFW_TYPE");
    }
    _row = -1;
    _nRows = _fits->countRows();
    table.preallocate(_nRows);
}

PTR(BaseTable) FitsReader::_readTable() {
    PTR(daf::base::PropertyList) metadata = boost::make_shared<daf::base::PropertyList>();
    _fits->readMetadata(*metadata, true);
    Schema schema;
    _fields = _readSchema(schema, *metadata, true);
    PTR(BaseTable) table = BaseTable::make(schema);
    table->setMetadata(metadata);
    _startRecords(*table);
    return table;
}

PTR(BaseRecord) FitsReader::_readRecord(PTR(BaseTable) const & table) {

    // Create empty record, return it if we're at the end already.
    PTR(BaseRecord) record;
    if (++_row == _nRows) return record;
    record = table->makeRecord();

    // Read all flags into an array of bool.
    // Most of this should be done once per table, not once per record.
    _fits->behavior &= ~ Fits::AUTO_CHECK; // temporarily disable automatic FITS exceptions
    int flagCol = -1;
    _fits->readKey("FLAGCOL", flagCol);
    boost::scoped_array<bool> flagData;
    if (_fits->status == 0) {
        --flagCol; // we want 0-indexed column numbers, not FITS' 1-indexed numbers
        int nFlags = _fits->getTableArraySize(flagCol);
        if (nFlags) {
            flagData.reset(new bool[nFlags]);
            _fits->readTableArray<bool>(_row, flagCol, nFlags, flagData.get());
        }
    } else {
        _fits->status = 0;
        flagCol = -1;
    }
    _fits->behavior |= Fits::AUTO_CHECK;

    // Read fields into the record.
    for (auto iter = _fields.begin(); iter != _fields.end(); ++iter) {
        (**iter).readCell(_row, *record, *_fits, flagData);
    }
    return record;
}

// ------------ FitsReader Registry implementation ----------------------------------------------------------

namespace {

typedef std::map<std::string,FitsReader::Factory*> Registry;

Registry & getRegistry() {
    static Registry it;
    return it;
}

// here's an example of how you register a FitsReader
static FitsReader::FactoryT<FitsReader> baseReaderFactory("BASE");

} // anonymous

FitsReader::Factory::Factory(std::string const & name) {
    getRegistry()[name] = this;
}

FitsReader::~FitsReader() {}

PTR(FitsReader) FitsReader::make(Fits * fits, PTR(io::InputArchive) archive, int flags) {
    std::string name;
    fits->behavior &= ~Fits::AUTO_CHECK; // temporarily disable automatic FITS exceptions
    fits->readKey("AFW_TYPE", name);
    if (fits->status != 0) {
        name = "BASE";
        fits->status = 0;
    }
    fits->behavior |= Fits::AUTO_CHECK;
    Registry::iterator i = getRegistry().find(name);
    if (i == getRegistry().end()) {
        throw LSST_EXCEPT(
            lsst::pex::exceptions::NotFoundError,
            (boost::format("FitsReader with name '%s' does not exist; check AFW_TYPE keyword.") % name).str()
        );
    }
    return (*i->second)(fits, archive, flags);
}

}}}} // namespace lsst::afw::table::io
