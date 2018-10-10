import numpy.testing as npt
import handyspark
from handyspark.sql.schema import generate_schema

def test_generate_schema(sdf):
    res = sdf.schema
    hres = generate_schema(sdf.columns, sdf.toPandas().dtypes.values)
    print(res)
    print(hres)
    npt.assert_array_equal(hres, res)
