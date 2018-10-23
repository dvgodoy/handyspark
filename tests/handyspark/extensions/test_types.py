from handyspark import *
import numpy.testing as npt
from pyspark.sql.types import IntegerType, StringType, ArrayType, MapType

def test_atomic_types():
    npt.assert_equal(IntegerType.ret('')[1], 'integer')
    npt.assert_equal(StringType.ret('')[1], 'string')

def test_composite_types():
    npt.assert_equal(ArrayType(IntegerType()).ret('')[1], 'array<int>')
    npt.assert_equal(MapType(StringType(), IntegerType()).ret('')[1], 'map<string,int>')