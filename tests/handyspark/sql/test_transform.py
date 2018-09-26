import numpy.testing as npt
from handyspark.sql.transform import HandyTransform
from pyspark.sql.types import DoubleType, StringType, ArrayType

ht = HandyTransform()

def test_apply_axis0(sdf, pdf):
    hdf = sdf.toHandy
    # setting the type manually
    hres1 = ht.apply(hdf, lambda Fare: Fare.map('${:,.2f}'.format), 'new', returnType='string').handy['new', None]
    # setting the type using an extension
    hres2 = ht.apply(hdf, StringType.ret(lambda Fare: Fare.map('${:,.2f}'.format)), 'new').handy['new', None]
    res = pdf.Fare.map('${:,.2f}'.format)
    npt.assert_array_equal(hres1, res)
    npt.assert_array_equal(hres2, res)

def test_apply_axis1(sdf, pdf):
    hdf = sdf.toHandy
    # setting the type manually
    hres1 = ht.apply(hdf, lambda Fare, Age: Fare / Age, 'new', returnType='double').handy['new', None]
    # setting the type using an extension
    hres2 = ht.apply(hdf, DoubleType.ret(lambda Fare, Age: Fare / Age), 'new').handy['new', None]
    # inferring type from 1st argument
    hres3 = ht.apply(hdf, lambda Fare, Age: Fare / Age, 'new').handy['new', None]
    res = pdf.apply(lambda row: row.Fare / row.Age, axis=1)
    npt.assert_array_equal(hres1, res)
    npt.assert_array_equal(hres2, res)
    npt.assert_array_equal(hres3, res)

def test_transform_axis0(sdf, pdf):
    hdf = sdf.toHandy
    # setting the type manually
    hres1 = ht.transform(hdf, lambda Fare: Fare.map('${:,.2f}'.format), 'new', returnType='string').handy['new', None]
    # setting the type using an extension
    hres2 = ht.transform(hdf, StringType.ret(lambda Fare: Fare.map('${:,.2f}'.format)), 'new').handy['new', None]
    res = pdf.Fare.map('${:,.2f}'.format)
    npt.assert_array_equal(hres1, res)
    npt.assert_array_equal(hres2, res)

def test_transform_axis1(sdf, pdf):
    hdf = sdf.toHandy
    # setting the type manually
    hres1 = ht.transform(hdf, lambda Fare, Age: Fare / Age, 'new', returnType='double').handy['new', None]
    # setting the type using an extension
    hres2 = ht.transform(hdf, DoubleType.ret(lambda Fare, Age: Fare / Age), 'new').handy['new', None]
    # inferring type from 1st argument
    hres3 = ht.transform(hdf, lambda Fare, Age: Fare / Age, 'new').handy['new', None]
    res = pdf.apply(lambda row: row.Fare / row.Age, axis=1)
    npt.assert_array_equal(hres1, res)
    npt.assert_array_equal(hres2, res)
    npt.assert_array_equal(hres3, res)

def test_assign_axis0(sdf, pdf):
    hdf = sdf.toHandy
    # setting the type using an extension
    hres = ht.assign(hdf, new=StringType.ret(lambda Fare: Fare.map('${:,.2f}'.format))).handy['new', None]
    res = pdf.assign(new=pdf.Fare.map('${:,.2f}'.format))['new']
    npt.assert_array_equal(hres, res)

def test_assign_axis1(sdf, pdf):
    hdf = sdf.toHandy
    # inferring type from 1st argument
    hres1 = ht.assign(hdf, new=lambda Fare, Age: Fare / Age).handy['new', None]
    # setting the type using an extension
    hres2 = ht.assign(hdf, new=DoubleType.ret(lambda Fare, Age: Fare / Age)).handy['new', None]
    res = pdf.assign(new=pdf.Fare / pdf.Age)['new']
    npt.assert_array_almost_equal(hres1, res)
    npt.assert_array_almost_equal(hres2, res)
