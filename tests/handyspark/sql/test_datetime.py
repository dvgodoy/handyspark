import numpy.testing as npt
import handyspark

def test_is_leap_year(sdates, pdates):
    hdf = sdates.toHandy()
    hdf = hdf.assign(newcol=hdf.pandas['dates'].dt.is_leap_year)
    hres = hdf.cols['newcol'][:20]
    res = pdates['dates'].dt.is_leap_year[:20]
    npt.assert_array_equal(hres, res)

def test_strftime(sdates, pdates):
    hdf = sdates.toHandy()
    hdf = hdf.assign(newcol=hdf.pandas['dates'].dt.strftime(date_format='%Y-%m'))
    hres = hdf.cols['newcol'][:20]
    res = pdates['dates'].dt.strftime(date_format='%Y-%m')[:20]
    npt.assert_array_equal(hres, res)

def test_weekday_name(sdates, pdates):
    hdf = sdates.toHandy()
    hdf = hdf.assign(newcol=hdf.pandas['dates'].dt.weekday_name)
    hres = hdf.cols['newcol'][:20]
    res = pdates['dates'].dt.weekday_name[:20]
    npt.assert_array_equal(hres, res)

def test_round(sdates, pdates):
    hdf = sdates.toHandy()
    hdf = hdf.assign(newcol=hdf.pandas['dates'].dt.round(freq='D'))
    hres = hdf.cols['newcol'][:20]
    res = pdates['dates'].dt.round(freq='D')[:20]
    npt.assert_array_equal(hres, res)
