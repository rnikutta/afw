// -*- lsst-c++ -*-

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
 
/** @file
  * @ingroup afw
  */
#include <algorithm>
#include <cmath>

#include "boost/scoped_array.hpp"
#include "boost/iterator/transform_iterator.hpp"

#include "lsst/utils/ieee.h"
#include "lsst/pex/exceptions.h"
#include "lsst/pex/logging/Trace.h"
#include "lsst/afw/table/SourceMatch.h"
#include "lsst/afw/geom/Angle.h"

namespace lsst { namespace afw { namespace table { namespace {

struct SourcePos {
    double dec;
    double x;
    double y;
    double z;
    // JFB removed extra pointer here; this may have performance implications, but hopefully not
    // significant ones.  SourceVector iterators yield temporary SourceRecord PTRs, so storing
    // their address was no longer an option.
    PTR(SourceRecord) src;
};

bool operator<(SourcePos const &s1, SourcePos const &s2) {
    return (s1.dec < s2.dec);
}

struct CmpSourcePtr {
    bool operator()(PTR(SourceRecord) const s1, PTR(SourceRecord) const s2) {
        return s1->getY() < s2->getY();
    }
};

/**
 * Extract source positions from @a set, convert them to cartesian coordinates
 * (for faster distance checks) and sort the resulting array of @c SourcePos
 * instances by declination. Sources with positions containing a NaN are skipped.
 *
 * @param[in] set          set of sources to process
 * @param[out] positions   pointer to an array of at least @c set.size()
 *                         SourcePos instances
 * @return                 The number of sources with positions not containing a NaN.
 */
size_t makeSourcePositions(SourceVector const &set, SourcePos *positions) {
    size_t n = 0;
    for (SourceVector::const_iterator i(set.begin()), e(set.end()); i != e; ++i) {
        geom::Angle ra = i->getRa();
        geom::Angle dec = i->getDec();
        if (lsst::utils::isnan(ra.asRadians()) || lsst::utils::isnan(dec.asRadians())) {
            continue;
        }
        double cosDec    = std::cos(dec);
        positions[n].dec = dec.asRadians();
        positions[n].x   = std::cos(ra)*cosDec;
        positions[n].y   = std::sin(ra)*cosDec;
        positions[n].z   = std::sin(dec);
        positions[n].src = i;
        ++n;
    }
    std::sort(positions, positions + n);
    if (n < set.size()) {
        lsst::pex::logging::TTrace<1>("afw.table.matchRaDec",
                                      "At least one source had ra or dec equal to NaN");
    }
    return n;
}

} // <anonymous>


/** Compute all tuples (s1,s2,d) where s1 belings to @a set1, s2 belongs to @a set2 and
  * d, the distance between s1 and s2, is at most @a radius. If set1 and
  * set2 are identical, then this call is equivalent to @c matchRaDec(set1,radius,true).
  * The match is performed in ra, dec space.
  *
  * @param[in] set1     first set of sources
  * @param[in] set2     second set of sources
  * @param[in] radius   match radius
  * @param[in] closest  if true then just return the closest match
  */
std::vector<SourceMatch> matchRaDec(SourceVector const &set1,
                                    SourceVector const &set2,
                                    geom::Angle radius, bool closest) {
    if (&set1 == &set2) {
        return matchRaDec(set1, radius, true);
    }
    if (radius < 0.0 || (radius > (45. * geom::degrees))) {
        throw LSST_EXCEPT(pex::exceptions::RangeErrorException, 
                          "match radius out of range (0 to 45 degrees)");
    }
    if (set1.size() == 0 || set2.size() == 0) {
        return std::vector<SourceMatch>();
    }
    // setup match parameters
    double const d2Limit = radius.toUnitSphereDistanceSquared();

    // Build position lists
    size_t len1 = set1.size();
    size_t len2 = set2.size();
    boost::scoped_array<SourcePos> pos1(new SourcePos[len1]);
    boost::scoped_array<SourcePos> pos2(new SourcePos[len2]);
    len1 = makeSourcePositions(set1, pos1.get());
    len2 = makeSourcePositions(set2, pos2.get());

    std::vector<SourceMatch> matches;
    for (size_t i = 0, start = 0; i < len1; ++i) {
        double minDec = pos1[i].dec - radius.asRadians();
        while (start < len2 && pos2[start].dec < minDec) { ++start; }
        if (start == len2) {
            break;
        }
        double maxDec = pos1[i].dec + radius.asRadians();
        size_t closestIndex = -1;          // Index of closest match (if any)
        double d2Include = d2Limit;     // Squared distance for inclusion of match
        bool found = false;             // Found anything?
        for (size_t j = start; j < len2 && pos2[j].dec <= maxDec; ++j) {
            double dx = pos1[i].x - pos2[j].x;
            double dy = pos1[i].y - pos2[j].y;
            double dz = pos1[i].z - pos2[j].z;
            double d2 = dx*dx + dy*dy + dz*dz;
            if (d2 < d2Include) {
                if (closest) {
                    d2Include = d2;
                    closestIndex = j;
                    found = true;
                } else {
                    matches.push_back(
                        SourceMatch(pos1[i].src, pos2[j].src,
                                    geom::Angle::fromUnitSphereDistanceSquared(d2).asRadians())
                    );
                }
            }
        }
        if (closest && found) {
            matches.push_back(
                SourceMatch(pos1[i].src, pos2[closestIndex].src,
                            geom::Angle::fromUnitSphereDistanceSquared(d2Include).asRadians())
            );
        }
    }
    return matches;
}


/** Compute all tuples (s1,s2,d) where s1 != s2, s1 and s2 both belong to @a set,
  * and d, the distance between s1 and s2, is at most @a radius. The
  * match is performed in ra, dec space.
  *
  * @param[in] set          the set of sources to self-match
  * @param[in] radius       match radius
  * @param[in] symmetric    if set to @c true symmetric matches are produced: i.e.
  *                         if (s1, s2, d) is reported, then so is (s2, s1, d).
  */
std::vector<SourceMatch> matchRaDec(SourceVector const &set,
                                    geom::Angle radius,
                                    bool symmetric) {
    if (radius < 0.0 || radius > (45.0 * geom::degrees)) {
        throw LSST_EXCEPT(pex::exceptions::RangeErrorException,
                          "match radius out of range (0 to 45 degrees)");
    }
    if (set.size() == 0) {
        return std::vector<SourceMatch>();
    }
    // setup match parameters
    double const d2Limit = radius.toUnitSphereDistanceSquared();

    // Build position list
    size_t len = set.size();
    boost::scoped_array<SourcePos> pos(new SourcePos[len]);
    len = makeSourcePositions(set, pos.get());

    std::vector<SourceMatch> matches;
    for (size_t i = 0; i < len; ++i) {
        double maxDec = pos[i].dec + radius.asRadians();
        for (size_t j = i + 1; j < len && pos[j].dec <= maxDec; ++j) {
            double dx = pos[i].x - pos[j].x;
            double dy = pos[i].y - pos[j].y;
            double dz = pos[i].z - pos[j].z;
            double d2 = dx*dx + dy*dy + dz*dz;
            if (d2 < d2Limit) {
                geom::Angle d = geom::Angle::fromUnitSphereDistanceSquared(d2);
                matches.push_back(SourceMatch(pos[i].src, pos[j].src, d.asRadians()));
                if (symmetric) {
                    matches.push_back(SourceMatch(pos[j].src, pos[i].src, d.asRadians()));
                }
            }
        }
    }
    return matches;
}


/** Compute all tuples (s1,s2,d) where s1 belings to @a set1, s2 belongs to @a set2 and
  * d, the distance between s1 and s2, in pixels, is at most @a radius. If set1 and
  * set2 are identical, then this call is equivalent to @c matchXy(set1,radius,true).
  * The match is performed in pixel space (2d cartesian).
  *
  * @param[in] set1     first set of sources
  * @param[in] set2     second set of sources
  * @param[in] radius   match radius (pixels)
  * @param[in] closest  if true then just return the closest match
  */
std::vector<SourceMatch> matchXy(SourceVector const &set1,
                                 SourceVector const &set2,
                                 double radius, bool closest) {
    if (&set1 == &set2) {
       return matchXy(set1, radius);
    }
    // setup match parameters
    double const r2 = radius*radius;

    // copy and sort array of pointers on y
    size_t const len1 = set1.size();
    size_t const len2 = set2.size();
    boost::scoped_array<PTR(SourceRecord)> pos1(new PTR(SourceRecord)[len1]);
    boost::scoped_array<PTR(SourceRecord)> pos2(new PTR(SourceRecord)[len2]);
    size_t n = 0;
    for (SourceVector::const_iterator i(set1.begin()), e(set1.end()); i != e; ++i, ++n) {
        pos1[n] = i;
    }
    n = 0;
    for (SourceVector::const_iterator i(set2.begin()), e(set2.end()); i != e; ++i, ++n) {
        pos2[n] = i;
    }
    std::sort(pos1.get(), pos1.get() + len1, CmpSourcePtr());
    std::sort(pos2.get(), pos2.get() + len2, CmpSourcePtr());

    std::vector<SourceMatch> matches;
    for (size_t i = 0, start = 0; i < len1; ++i) {
        double y = pos1[i]->getY();
        double minY = y - radius;
        while (start < len2 && pos2[start]->getY() < minY) { ++start; }
        if (start == len2) {
            break;
        }
        double x = pos1[i]->getX();
        double maxY = y + radius;
        double y2;
        size_t closestIndex = -1;          // Index of closest match (if any)
        double r2Include = r2;          // Squared radius for inclusion of match
        bool found = false;             // Found anything?
        for (size_t j = start; j < len2 && (y2 = pos2[j]->getY()) <= maxY; ++j) {
            double dx = x - pos2[j]->getX();
            double dy = y - y2;
            double d2 = dx*dx + dy*dy;
            if (d2 < r2Include) {
                if (closest) {
                    r2Include = d2;
                    closestIndex = j;
                    found = true;
                } else {
                    matches.push_back(SourceMatch(pos1[i], pos2[j], std::sqrt(d2)));
                }
            }
        }
        if (closest && found) {
            matches.push_back(SourceMatch(pos1[i], pos2[closestIndex], std::sqrt(r2Include)));
        }
    }
    return matches;
}


/** Compute all tuples (s1,s2,d) where s1 != s2, s1 and s2 both belong to @a set,
  * and d, the distance between s1 and s2, in pixels, is at most @a radius. The
  * match is performed in pixel space (2d cartesian).
  *
  * @param[in] set          the set of sources to self-match
  * @param[in] radius       match radius (pixels)
  * @param[in] symmetric    if set to @c true symmetric matches are produced: i.e.
  *                         if (s1, s2, d) is reported, then so is (s2, s1, d).
  */
std::vector<SourceMatch> matchXy(SourceVector const &set, double radius, bool symmetric) {
    // setup match parameters
    double const r2 = radius*radius;

    // copy and sort array of pointers on y
    size_t const len = set.size();
    boost::scoped_array<PTR(SourceRecord)> pos(new PTR(SourceRecord)[len]);
    size_t n = 0;
    for (SourceVector::const_iterator i(set.begin()), e(set.end()); i != e; ++i, ++n) {
        pos[n] = i;
    }
    std::sort(pos.get(), pos.get() + len, CmpSourcePtr());

    std::vector<SourceMatch> matches;
    for (size_t i = 0; i < len; ++i) {
        double x = pos[i]->getX();
        double y = pos[i]->getY();
        double maxY = y + radius;
        double y2;
        for (size_t j = i + 1; j < len && (y2 = pos[j]->getY()) <= maxY; ++j) {
            double dx = x - pos[j]->getX();
            double dy = y - y2;
            double d2 = dx*dx + dy*dy;
            if (d2 < r2) {
                double d = std::sqrt(d2);
                matches.push_back(SourceMatch(pos[i], pos[j], d));
                if (symmetric) {
                    matches.push_back(SourceMatch(pos[j], pos[i], d));
                }
            }
        }
    }
    return matches;
}

BaseVector makeSourceMatchTable(SourceMatchVector const & matches) {
    Schema schema;
    Key<RecordId> key1 = schema.addField<RecordId>("first", "ID for first source record in match.");
    Key<RecordId> key2 = schema.addField<RecordId>("second", "ID for second source record in match.");
    Key<double> keyD = schema.addField<double>("distance", "Distance between matches sources.",
                                               "pixels or radians");
    BaseVector result(schema);
    result.getTable()->preallocate(matches.size());
    result.reserve(matches.size());
    for (SourceMatchVector::const_iterator i = matches.begin(); i != matches.end(); ++i) {
        PTR(BaseRecord) record = result.addNew();
        record->set(key1, i->first->getId());
        record->set(key2, i->second->getId());
        record->set(keyD, i->distance);
    }
    return result;
}

namespace {

struct ExtractId {
    typedef RecordId result_type;
    typedef SourceRecord const & argument_type;
    RecordId operator()(SourceRecord const & s) const { return s.getId(); } 
};

PTR(SourceRecord) findSourceById(SourceVector const & vector, RecordId id) {
    
    // Iterator type that makes a SourceVector iterator look like an iterator over IDs.
    typedef boost::transform_iterator<ExtractId,SourceVector::const_iterator> SearchIter;

    SearchIter i = std::lower_bound(SearchIter(vector.begin()), SearchIter(vector.end()), id);
    if (i.base() != vector.end() && *i == id) return i.base();
    return PTR(SourceRecord)();
}


} // anonymous

SourceMatchVector makeSourceMatchVector(
    BaseVector const & matches, 
    SourceVector const & first,
    SourceVector const & second
) {
    Key<RecordId> key1 = matches.getSchema()["first"];
    Key<RecordId> key2 = matches.getSchema()["second"];
    Key<double> keyD = matches.getSchema()["distance"];
    SourceMatchVector result;
    result.resize(matches.size());
    SourceMatchVector::iterator j = result.begin();
    for (BaseVector::const_iterator i = matches.begin(); i != matches.end(); ++i, ++j) {
        j->first = findSourceById(first, i->get(key1));
        j->second = findSourceById(second, i->get(key2));
        j->distance = i->get(keyD);
    }
    return result;
}

}}} // namespace lsst::afw::table