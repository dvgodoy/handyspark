import numpy.testing as npt
import handyspark

# boolean returns
def test_between(sdf, pdf):
    hdf = sdf.toHandy
    hres = hdf.pandas.between('Age', left=20, right=40, alias='newcol').handy['newcol']
    res = pdf['Age'].between(left=20, right=40)[:20]
    npt.assert_array_equal(hres, res)

def test_isin(sdf, pdf):
    hdf = sdf.toHandy
    hres = hdf.pandas.isin('Age', values=[22, 40], alias='newcol').handy['newcol']
    res = pdf['Age'].isin(values=[22, 40])[:20]
    npt.assert_array_equal(hres, res)

def test_isna(sdf, pdf):
    hdf = sdf.toHandy
    hres = hdf.pandas.isna('Cabin', alias='newcol').handy['newcol']
    res = pdf['Cabin'].isna()[:20]
    npt.assert_array_equal(hres, res)

def test_notna(sdf, pdf):
    hdf = sdf.toHandy
    hres = hdf.pandas.notna('Cabin', alias='newcol').handy['newcol']
    res = pdf['Cabin'].notna()[:20]
    npt.assert_array_equal(hres, res)

# same type returns
def test_clip(sdf, pdf):
    hdf = sdf.toHandy
    hres = hdf.pandas.clip('Age', lower=5, upper=50, alias='newcol').handy['newcol']
    res = pdf['Age'].clip(lower=5, upper=50)[:20]
    npt.assert_array_equal(hres, res)

def test_replace(sdf, pdf):
    hdf = sdf.toHandy
    hres = hdf.pandas.replace('Age', to_replace=5, value=0, alias='newcol').handy['newcol']
    res = pdf['Age'].replace(to_replace=5, value=0)[:20]
    npt.assert_array_equal(hres, res)

def test_round(sdf, pdf):
    hdf = sdf.toHandy
    hres = hdf.pandas.round('Fare', decimals=0, alias='newcol').handy['newcol']
    res = pdf['Fare'].round(decimals=0)[:20]
    npt.assert_array_equal(hres, res)
