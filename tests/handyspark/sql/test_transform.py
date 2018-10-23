import numpy.testing as npt
from pyspark.sql.types import DoubleType, StringType
from handyspark import *

def test_apply_axis0(sdf, pdf):
    hdf = sdf.toHandy()
    # setting the type manually
    hres1 = hdf.apply(lambda Fare: Fare.map('${:,.2f}'.format), 'new', returnType='string').cols['new'][:]
    # setting the type using an extension
    hres2 = hdf.apply(StringType.ret(lambda Fare: Fare.map('${:,.2f}'.format)), 'new').cols['new'][:]
    res = pdf.Fare.map('${:,.2f}'.format)
    npt.assert_array_equal(hres1, res)
    npt.assert_array_equal(hres2, res)

def test_apply_axis1(sdf, pdf):
    hdf = sdf.toHandy()
    # setting the type manually
    hres1 = hdf.apply(lambda Fare, Age: Fare / Age, 'new', returnType='double').cols['new'][:]
    # setting the type using an extension
    hres2 = hdf.apply(DoubleType.ret(lambda Fare, Age: Fare / Age), 'new').cols['new'][:]
    # inferring type from 1st argument
    hres3 = hdf.apply(lambda Fare, Age: Fare / Age, 'new').cols['new'][:]
    res = pdf.apply(lambda row: row.Fare / row.Age, axis=1)
    npt.assert_array_equal(hres1, res)
    npt.assert_array_equal(hres2, res)
    npt.assert_array_equal(hres3, res)

def test_transform_axis0(sdf, pdf):
    hdf = sdf.toHandy()
    # setting the type manually
    hres1 = hdf.transform(lambda Fare: Fare.map('${:,.2f}'.format), 'new', returnType='string').cols['new'][:]
    # setting the type using an extension
    hres2 = hdf.transform(StringType.ret(lambda Fare: Fare.map('${:,.2f}'.format)), 'new').cols['new'][:]
    res = pdf.Fare.map('${:,.2f}'.format)
    npt.assert_array_equal(hres1, res)
    npt.assert_array_equal(hres2, res)

def test_transform_axis1(sdf, pdf):
    hdf = sdf.toHandy()
    # setting the type manually
    hres1 = hdf.transform(lambda Fare, Age: Fare / Age, 'new', returnType='double').cols['new'][:]
    # setting the type using an extension
    hres2 = hdf.transform(DoubleType.ret(lambda Fare, Age: Fare / Age), 'new').cols['new'][:]
    # inferring type from 1st argument
    hres3 = hdf.transform(lambda Fare, Age: Fare / Age, 'new').cols['new'][:]
    res = pdf.apply(lambda row: row.Fare / row.Age, axis=1)
    npt.assert_array_equal(hres1, res)
    npt.assert_array_equal(hres2, res)
    npt.assert_array_equal(hres3, res)

def test_assign_axis0(sdf, pdf):
    hdf = sdf.toHandy()
    # setting the type using an extension
    hres = hdf.assign(new=StringType.ret(lambda Fare: Fare.map('${:,.2f}'.format))).cols['new'][:]
    res = pdf.assign(new=pdf.Fare.map('${:,.2f}'.format))['new']
    npt.assert_array_equal(hres, res)

def test_assign_axis1(sdf, pdf):
    hdf = sdf.toHandy()
    # inferring type from 1st argument
    hres1 = hdf.assign(new=lambda Fare, Age: Fare / Age).cols['new'][:]
    # setting the type using an extension
    hres2 = hdf.assign(new=DoubleType.ret(lambda Fare, Age: Fare / Age)).cols['new'][:]
    res = pdf.assign(new=pdf.Fare / pdf.Age)['new']
    npt.assert_array_almost_equal(hres1, res)
    npt.assert_array_almost_equal(hres2, res)
