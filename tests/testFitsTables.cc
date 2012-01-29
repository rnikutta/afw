#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE table-fits
#include "boost/test/unit_test.hpp"

#include <iostream>
#include <iterator>
#include <algorithm>
#include <map>

#include "lsst/utils/ieee.h"
#include "lsst/afw/table/Source.h"

struct EqualityCompare {

    bool operator()(
        lsst::afw::detection::Span::Ptr const & a, 
        lsst::afw::detection::Span::Ptr const & b
    ) const {
        return a->getY() == b->getY() && a->getX0() == b->getX0() && a->getX1() == b->getX1();
    }

    bool operator()(float a, float b) const {
        return std::fabs(a - b) < 1E-8 || (lsst::utils::isnan(a) && lsst::utils::isnan(b));
    }

    bool operator()(
        lsst::afw::detection::Peak::Ptr const & a, 
        lsst::afw::detection::Peak::Ptr const & b
    ) const {
        return (*this)(a->getFx(), b->getFx())
            && (*this)(a->getFy(), b->getFy())
            && (*this)(a->getPeakValue(), b->getPeakValue());
    }

};


BOOST_AUTO_TEST_CASE(testFits) {
    using namespace lsst::afw::table;

    Schema schema = SourceTable::makeMinimalSchema();
    Key<int> a_b_i = schema.addField<int>("a.b.i", "int");
    Key<Flag> a_b_i_valid = schema.addField<Flag>("a.b.i.valid", "is field a.b.i valid?");
    Key<float> a_c_f = schema.addField<float>("a.c.f", "float", "femtoseamonkeys");
    Key<double> e_g_d = schema.addField<double>("e.g.d", "double", "bargles^2");
    Key<Flag> e_g_d_flag1 = schema.addField<Flag>("e.g.d.flag1", "flag1 for e.g.d");
    Key<Flag> e_g_d_flag2 = schema.addField<Flag>("e.g.d.flag2", "flag2 for e.g.d");
    Key< Point<float> > a_b_p = schema.addField< Point<float> >("a.b.p", "point", "pixels");

    Key<double> flux = schema.addField<double>("flux", "flux doc");
    Key<double> fluxErr = schema.addField<double>("flux.err", "flux err doc");
    Key< Point<double> > centroid = schema.addField< Point<double> >("centroid", "centroid doc");
    Key< Covariance< Point<double> > > centroidCov = schema.addField< Covariance< Point<double> > >(
        "centroid.cov", "centroid covariance doc"
    );

    SourceVector vector(SourceTable::make(schema));

    vector.getTable()->defineModelFlux(flux, fluxErr);
    vector.getTable()->defineCentroid(centroid, centroidCov);

    {
        PTR(Footprint) fp1 = boost::make_shared<Footprint>();
        fp1->addSpan(0, 5, 8);
        fp1->addSpan(1, 4, 9);
        fp1->addSpan(2, 6, 7);
        fp1->getPeaks().push_back(boost::make_shared<lsst::afw::detection::Peak>(4.5f, 1.2f, 25.6f));
        fp1->getPeaks().push_back(boost::make_shared<lsst::afw::detection::Peak>(6.8f, 0.8f, 23.2f));
        PTR(SourceRecord) r1 = vector.getTable()->makeRecord();
        r1->setFootprint(fp1);
        
        r1->set(a_b_i, 314);
        r1->set(a_b_i_valid, true);
        r1->set(a_c_f, 3.14f);
        r1->set(e_g_d, 3.14E12);
        r1->set(e_g_d_flag1, false);
        r1->set(e_g_d_flag2, true);
        r1->set(a_b_p, lsst::afw::geom::Point2D(1.2, 0.5));
        vector.push_back(r1);

        PTR(SourceRecord) r2 = vector.getTable()->makeRecord();
        r2->set(a_b_i, 5123);
        r2->set(a_b_i_valid, true);
        r2->set(a_c_f, 44.8f);
        r2->set(e_g_d, 12.2E-3);
        r2->set(e_g_d_flag1, true);
        r2->set(e_g_d_flag2, false);
        r2->set(a_b_p, lsst::afw::geom::Point2D(-32.1, 63.2));
        PTR(Footprint) fp2 = boost::make_shared<Footprint>();
        fp2->addSpan(3, 2, 7);
        fp2->addSpan(4, 3, 5);
        fp2->getPeaks().push_back(boost::make_shared<lsst::afw::detection::Peak>(4.2f, 3.3f, 32.1f));
        r2->setFootprint(fp2);
        vector.push_back(r2);

        BOOST_CHECK_EQUAL( r2->get(e_g_d_flag1), true );
        BOOST_CHECK_EQUAL( r2->get(e_g_d_flag2), false );
    }

    vector.writeFits("!testTable.fits");

    SourceVector readVector = SourceVector::readFits("testTable.fits[1]");
    BOOST_CHECK_EQUAL( schema, readVector.getSchema() );

    BOOST_CHECK_EQUAL( vector.getTable()->getModelFluxKey(), readVector.getTable()->getModelFluxKey() );
    BOOST_CHECK_EQUAL( vector.getTable()->getModelFluxErrKey(), readVector.getTable()->getModelFluxErrKey() );

    BOOST_CHECK_EQUAL( vector.getTable()->getCentroidKey(), readVector.getTable()->getCentroidKey() );
    BOOST_CHECK_EQUAL( vector.getTable()->getCentroidCovKey(), readVector.getTable()->getCentroidCovKey() );

    {
        SourceRecord const & a1 = vector[0];
        SourceRecord const & b1 = readVector[0];
        BOOST_CHECK_EQUAL( a1.get(a_b_i), b1.get(a_b_i) );
        BOOST_CHECK_EQUAL( a1.get(a_b_i_valid), b1.get(a_b_i_valid) );
        BOOST_CHECK_CLOSE_FRACTION( a1.get(a_c_f), b1.get(a_c_f), 1E-8 );
        BOOST_CHECK_CLOSE_FRACTION( a1.get(e_g_d), b1.get(e_g_d), 1E-16 );
        BOOST_CHECK_EQUAL( a1.get(e_g_d_flag1), b1.get(e_g_d_flag1) );
        BOOST_CHECK_EQUAL( a1.get(e_g_d_flag2), b1.get(e_g_d_flag2) );
        BOOST_CHECK_CLOSE_FRACTION( a1.get(a_b_p.getX()), b1.get(a_b_p.getX()), 1E-8 );
        BOOST_CHECK_CLOSE_FRACTION( a1.get(a_b_p.getY()), b1.get(a_b_p.getY()), 1E-8 );
        Footprint const & fp1a = *a1.getFootprint();
        Footprint const & fp1b = *b1.getFootprint();
        BOOST_CHECK( std::equal(fp1a.getSpans().begin(), fp1a.getSpans().end(), fp1b.getSpans().begin(),
                                EqualityCompare()) );
        BOOST_CHECK( std::equal(fp1a.getPeaks().begin(), fp1a.getPeaks().end(), fp1b.getPeaks().begin(),
                                EqualityCompare()) );
        BOOST_CHECK_EQUAL( fp1a.getBBox(), fp1b.getBBox() );

        SourceRecord const & a2 = vector[1];
        SourceRecord const & b2 = readVector[1];
        BOOST_CHECK_EQUAL( a2.get(a_b_i), b2.get(a_b_i) );
        BOOST_CHECK_EQUAL( a2.get(a_b_i_valid), b2.get(a_b_i_valid) );
        BOOST_CHECK_CLOSE_FRACTION( a2.get(a_c_f), b2.get(a_c_f), 1E-8 );
        BOOST_CHECK_CLOSE_FRACTION( a2.get(e_g_d), b2.get(e_g_d), 1E-16 );
        BOOST_CHECK_EQUAL( a2.get(e_g_d_flag1), b2.get(e_g_d_flag1) );
        BOOST_CHECK_EQUAL( a2.get(e_g_d_flag2), b2.get(e_g_d_flag2) );
        BOOST_CHECK_CLOSE_FRACTION( a2.get(a_b_p.getX()), b2.get(a_b_p.getX()), 1E-8 );
        BOOST_CHECK_CLOSE_FRACTION( a2.get(a_b_p.getY()), b2.get(a_b_p.getY()), 1E-8 );
        Footprint const & fp2a = *a2.getFootprint();
        Footprint const & fp2b = *b2.getFootprint();
        BOOST_CHECK( std::equal(fp2a.getSpans().begin(), fp2a.getSpans().end(), fp2b.getSpans().begin(),
                                EqualityCompare()) );
        BOOST_CHECK( std::equal(fp2a.getPeaks().begin(), fp2a.getPeaks().end(), fp2b.getPeaks().begin(),
                                EqualityCompare()) );
        BOOST_CHECK_EQUAL( fp2a.getBBox(), fp2b.getBBox() );
    }
}