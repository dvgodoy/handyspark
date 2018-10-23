import numpy.testing as npt
from handyspark import *

# integer returns
def test_count(sdf, pdf):
    hdf = sdf.toHandy()
    hdf = hdf.assign(newcol=hdf.pandas['Name'].str.count(pat='Mr.'))
    hres = hdf.cols['newcol'][:20]
    res = pdf['Name'].str.count(pat='Mr.')[:20]
    npt.assert_array_equal(hres, res)

def test_find(sdf, pdf):
    hdf = sdf.toHandy()
    hdf = hdf.assign(newcol=hdf.pandas['Name'].str.find(sub='Mr.'))
    hres = hdf.cols['newcol'][:20]
    res = pdf['Name'].str.find(sub='Mr.')[:20]
    npt.assert_array_equal(hres, res)

def test_len(sdf, pdf):
    hdf = sdf.toHandy()
    hdf = hdf.assign(newcol=hdf.pandas['Name'].str.len())
    hres = hdf.cols['newcol'][:20]
    res = pdf['Name'].str.len()[:20]
    npt.assert_array_equal(hres, res)

# boolean returns
def test_rfind(sdf, pdf):
    hdf = sdf.toHandy()
    hdf = hdf.assign(newcol=hdf.pandas['Name'].str.rfind(sub='Mr.'))
    hres = hdf.cols['newcol'][:20]
    res = pdf['Name'].str.rfind(sub='Mr.')[:20]
    npt.assert_array_equal(hres, res)

def test_contains(sdf, pdf):
    hdf = sdf.toHandy()
    hdf = hdf.assign(newcol=hdf.pandas['Name'].str.contains(pat='Mr.'))
    hres = hdf.cols['newcol'][:20]
    res = pdf['Name'].str.contains(pat='Mr.')[:20]
    npt.assert_array_equal(hres, res)

def test_startswith(sdf, pdf):
    hdf = sdf.toHandy()
    hdf = hdf.assign(newcol=hdf.pandas['Name'].str.startswith(pat='Mr.'))
    hres = hdf.cols['newcol'][:20]
    res = pdf['Name'].str.startswith(pat='Mr.')[:20]
    npt.assert_array_equal(hres, res)

def test_match(sdf, pdf):
    hdf = sdf.toHandy()
    hdf = hdf.assign(newcol=hdf.pandas['Name'].str.match(pat='Mr.'))
    hres = hdf.cols['newcol'][:20]
    res = pdf['Name'].str.match(pat='Mr.')[:20]
    npt.assert_array_equal(hres, res)

def test_isalpha(sdf, pdf):
    hdf = sdf.toHandy()
    hdf = hdf.assign(newcol=hdf.pandas['Name'].str.isalpha())
    hres = hdf.cols['newcol'][:20]
    res = pdf['Name'].str.isalpha()[:20]
    npt.assert_array_equal(hres, res)

# string returns
def test_replace(sdf, pdf):
    hdf = sdf.toHandy()
    hdf = hdf.assign(newcol=hdf.pandas['Name'].str.replace(pat='Mr.', repl='Mister'))
    hres = hdf.cols['newcol'][:20]
    res = pdf['Name'].str.replace(pat='Mr.', repl='Mister')[:20]
    npt.assert_array_equal(hres, res)

def test_repeat(sdf, pdf):
    hdf = sdf.toHandy()
    hdf = hdf.assign(newcol=hdf.pandas['Name'].str.repeat(repeats=2))
    hres = hdf.cols['newcol'][:20]
    res = pdf['Name'].str.repeat(repeats=2)[:20]
    npt.assert_array_equal(hres, res)

def test_join(sdf, pdf):
    hdf = sdf.toHandy()
    hdf = hdf.assign(newcol=hdf.pandas['Name'].str.join(sep=','))
    hres = hdf.cols['newcol'][:20]
    res = pdf['Name'].str.join(sep=',')[:20]
    npt.assert_array_equal(hres, res)

def test_pad(sdf, pdf):
    hdf = sdf.toHandy()
    hdf = hdf.assign(newcol=hdf.pandas['Name'].str.pad(width=20))
    hres = hdf.cols['newcol'][:20]
    res = pdf['Name'].str.pad(width=20)[:20]
    npt.assert_array_equal(hres, res)

def test_slice(sdf, pdf):
    hdf = sdf.toHandy()
    hdf = hdf.assign(newcol=hdf.pandas['Name'].str.slice(start=5, stop=10))
    hres = hdf.cols['newcol'][:20]
    res = pdf['Name'].str.slice(start=5, stop=10)[:20]
    npt.assert_array_equal(hres, res)

def test_slice_replace(sdf, pdf):
    hdf = sdf.toHandy()
    hdf = hdf.assign(newcol=hdf.pandas['Name'].str.slice_replace(start=5, stop=10, repl='X'))
    hres = hdf.cols['newcol'][:20]
    res = pdf['Name'].str.slice_replace(start=5, stop=10, repl='X')[:20]
    npt.assert_array_equal(hres, res)

def test_strip(sdf, pdf):
    hdf = sdf.toHandy()
    hdf = hdf.assign(newcol=hdf.pandas['Name'].str.strip())
    hres = hdf.cols['newcol'][:20]
    res = pdf['Name'].str.strip()[:20]
    npt.assert_array_equal(hres, res)

def test_wrap(sdf, pdf):
    hdf = sdf.toHandy()
    hdf = hdf.assign(newcol=hdf.pandas['Name'].str.wrap(width=5))
    hres = hdf.cols['newcol'][:20]
    res = pdf['Name'].str.wrap(width=5)[:20]
    npt.assert_array_equal(hres, res)

def test_get(sdf, pdf):
    hdf = sdf.toHandy()
    hdf = hdf.assign(newcol=hdf.pandas['Name'].str.get(i=5))
    hres = hdf.cols['newcol'][:20]
    res = pdf['Name'].str.get(i=5)[:20]
    npt.assert_array_equal(hres, res)

def test_center(sdf, pdf):
    hdf = sdf.toHandy()
    hdf = hdf.assign(newcol=hdf.pandas['Name'].str.center(width=10))
    hres = hdf.cols['newcol'][:20]
    res = pdf['Name'].str.center(width=10)[:20]
    npt.assert_array_equal(hres, res)

def test_zfill(sdf, pdf):
    hdf = sdf.toHandy()
    hdf = hdf.assign(newcol=hdf.pandas['Name'].str.zfill(width=20))
    hres = hdf.cols['newcol'][:20]
    res = pdf['Name'].str.zfill(width=20)[:20]
    npt.assert_array_equal(hres, res)

def test_normalize(sdf, pdf):
    hdf = sdf.toHandy()
    hdf = hdf.assign(newcol=hdf.pandas['Name'].str.normalize(form='NFKD'))
    hres = hdf.cols['newcol'][:20]
    res = pdf['Name'].str.normalize(form='NFKD')[:20]
    npt.assert_array_equal(hres, res)

def test_upper(sdf, pdf):
    hdf = sdf.toHandy()
    hdf = hdf.assign(newcol=hdf.pandas['Name'].str.upper())
    hres = hdf.cols['newcol'][:20]
    res = pdf['Name'].str.upper()[:20]
    npt.assert_array_equal(hres, res)
