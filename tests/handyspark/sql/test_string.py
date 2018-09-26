import numpy.testing as npt
from handyspark.sql.string import HandyString

def test_find(sdf, pdf):
    hdf = sdf.toHandy
    hres = HandyString(hdf).find('Name', sub='Mr.', alias='FindMr').handy['FindMr']
    res = pdf['Name'].str.find(sub='Mr.')[:20]
    npt.assert_array_equal(hres, res)
