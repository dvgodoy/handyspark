import numpy.testing as npt
from pyspark.ml.feature import VectorAssembler
from handyspark.util import dense_to_array, disassemble

def test_dense_to_array(sdf):
    assem = VectorAssembler(inputCols=['Pclass', 'Fare', 'Age'], outputCol='features')
    tdf = assem.transform(sdf.dropna())
    tdf = dense_to_array(tdf, 'features', 'array_features')

    npt.assert_array_equal(tdf.series['features'][:], tdf.series['array_features'][:])

def test_disassemble(sdf):
    assem = VectorAssembler(inputCols=['Pclass', 'Fare', 'Age'], outputCol='features')
    tdf = assem.transform(sdf.dropna())
    tdf = disassemble(tdf, 'features')

    npt.assert_array_equal(tdf.series['Pclass'][:], tdf.series['features_0'][:])
    npt.assert_array_equal(tdf.series['Fare'][:], tdf.series['features_1'][:])
    npt.assert_array_equal(tdf.series['Age'][:], tdf.series['features_2'][:])
