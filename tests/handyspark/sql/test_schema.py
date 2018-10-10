import numpy as np
import numpy.testing as npt
import handyspark
from handyspark.sql.schema import generate_schema

def test_generate_schema(sdf):
    res = sdf.schema
    hres = generate_schema(sdf.columns, [np.int32, np.int32, np.int32, str, str, np.float64, np.int32, np.int32, str,
                                         np.float64, str, str])
    npt.assert_array_equal(hres, res)
