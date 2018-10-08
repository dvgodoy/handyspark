import numpy.testing as npt
import handyspark

# integer returns
def test_count(sdf, pdf):
    hdf = sdf.toHandy
    hres = hdf.pandas.str.count('Name', pat='Mr.', alias='newcol').handy['newcol']
    res = pdf['Name'].str.count(pat='Mr.')[:20]
    npt.assert_array_equal(hres, res)

def test_find(sdf, pdf):
    hdf = sdf.toHandy
    hres = hdf.pandas.str.find('Name', sub='Mr.', alias='newcol').handy['newcol']
    res = pdf['Name'].str.find(sub='Mr.')[:20]
    npt.assert_array_equal(hres, res)

def test_len(sdf, pdf):
    hdf = sdf.toHandy
    hres = hdf.pandas.str.len('Name', alias='newcol').handy['newcol']
    res = pdf['Name'].str.len()[:20]
    npt.assert_array_equal(hres, res)

# boolean returns
def test_rfind(sdf, pdf):
    hdf = sdf.toHandy
    hres = hdf.pandas.str.rfind('Name', sub='Mr.', alias='newcol').handy['newcol']
    res = pdf['Name'].str.rfind(sub='Mr.')[:20]
    npt.assert_array_equal(hres, res)

def test_contains(sdf, pdf):
    hdf = sdf.toHandy
    hres = hdf.pandas.str.contains('Name', pat='Mr.', alias='newcol').handy['newcol']
    res = pdf['Name'].str.contains(pat='Mr.')[:20]
    npt.assert_array_equal(hres, res)

def test_startswith(sdf, pdf):
    hdf = sdf.toHandy
    hres = hdf.pandas.str.startswith('Name', pat='Mr.', alias='newcol').handy['newcol']
    res = pdf['Name'].str.startswith(pat='Mr.')[:20]
    npt.assert_array_equal(hres, res)

def test_match(sdf, pdf):
    hdf = sdf.toHandy
    hres = hdf.pandas.str.match('Name', pat='Mr.', alias='newcol').handy['newcol']
    res = pdf['Name'].str.match(pat='Mr.')[:20]
    npt.assert_array_equal(hres, res)

def test_isalpha(sdf, pdf):
    hdf = sdf.toHandy
    hres = hdf.pandas.str.isalpha('Name', alias='newcol').handy['newcol']
    res = pdf['Name'].str.isalpha()[:20]
    npt.assert_array_equal(hres, res)

# string returns
def test_replace(sdf, pdf):
    hdf = sdf.toHandy
    hres = hdf.pandas.str.replace('Name', pat='Mr.', repl='Mister', alias='newcol').handy['newcol']
    res = pdf['Name'].str.replace(pat='Mr.', repl='Mister')[:20]
    npt.assert_array_equal(hres, res)

def test_repeat(sdf, pdf):
    hdf = sdf.toHandy
    hres = hdf.pandas.str.repeat('Name', repeats=2, alias='newcol').handy['newcol']
    res = pdf['Name'].str.repeat(repeats=2)[:20]
    npt.assert_array_equal(hres, res)

def test_join(sdf, pdf):
    hdf = sdf.toHandy
    hres = hdf.pandas.str.join('Name', sep=',', alias='newcol').handy['newcol']
    res = pdf['Name'].str.join(sep=',')[:20]
    npt.assert_array_equal(hres, res)

def test_pad(sdf, pdf):
    hdf = sdf.toHandy
    hres = hdf.pandas.str.pad('Name', width=20, alias='newcol').handy['newcol']
    res = pdf['Name'].str.pad(width=20)[:20]
    npt.assert_array_equal(hres, res)

def test_slice(sdf, pdf):
    hdf = sdf.toHandy
    hres = hdf.pandas.str.slice('Name', start=5, stop=10, alias='newcol').handy['newcol']
    res = pdf['Name'].str.slice(start=5, stop=10)[:20]
    npt.assert_array_equal(hres, res)

def test_slice_replace(sdf, pdf):
    hdf = sdf.toHandy
    hres = hdf.pandas.str.slice_replace('Name', start=5, stop=10, repl='X', alias='newcol').handy['newcol']
    res = pdf['Name'].str.slice_replace(start=5, stop=10, repl='X')[:20]
    npt.assert_array_equal(hres, res)

def test_strip(sdf, pdf):
    hdf = sdf.toHandy
    hres = hdf.pandas.str.strip('Name', alias='newcol').handy['newcol']
    res = pdf['Name'].str.strip()[:20]
    npt.assert_array_equal(hres, res)

def test_wrap(sdf, pdf):
    hdf = sdf.toHandy
    hres = hdf.pandas.str.wrap('Name', width=5, alias='newcol').handy['newcol']
    res = pdf['Name'].str.wrap(width=5)[:20]
    npt.assert_array_equal(hres, res)

def test_get(sdf, pdf):
    hdf = sdf.toHandy
    hres = hdf.pandas.str.get('Name', i=5, alias='newcol').handy['newcol']
    res = pdf['Name'].str.get(i=5)[:20]
    npt.assert_array_equal(hres, res)

def test_center(sdf, pdf):
    hdf = sdf.toHandy
    hres = hdf.pandas.str.center('Name', width=10, alias='newcol').handy['newcol']
    res = pdf['Name'].str.center(width=10)[:20]
    npt.assert_array_equal(hres, res)

def test_zfill(sdf, pdf):
    hdf = sdf.toHandy
    hres = hdf.pandas.str.zfill('Name', width=20, alias='newcol').handy['newcol']
    res = pdf['Name'].str.zfill(width=20)[:20]
    npt.assert_array_equal(hres, res)

def test_normalize(sdf, pdf):
    hdf = sdf.toHandy
    hres = hdf.pandas.str.normalize('Name', form='NFKD', alias='newcol').handy['newcol']
    res = pdf['Name'].str.normalize(form='NFKD')[:20]
    npt.assert_array_equal(hres, res)

def test_upper(sdf, pdf):
    hdf = sdf.toHandy
    hres = hdf.pandas.str.upper('Name', alias='newcol').handy['newcol']
    res = pdf['Name'].str.upper()[:20]
    npt.assert_array_equal(hres, res)
