import numpy as np
import numpy.testing as npt
from handyspark.sql import generate_schema

def test_generate_schema(sdf):
    res = sdf.select(sorted(sdf.columns)).schema
    hres = generate_schema(dict(zip(sdf.columns,
                                    [np.int32, np.int32, np.int32, str, str, np.float64,
                                     np.int32, np.int32, str, np.float64, str, str])))
    npt.assert_array_equal(hres, res)
