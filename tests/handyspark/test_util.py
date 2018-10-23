import numpy.testing as npt
from pyspark.ml.feature import VectorAssembler
from handyspark.util import dense_to_array, disassemble

def test_dense_to_array(sdf):
    assem = VectorAssembler(inputCols=['Pclass', 'Fare', 'Age'], outputCol='features')
    tdf = assem.transform(sdf.dropna())
    tdf = dense_to_array(tdf, 'features', 'array_features')

    npt.assert_array_equal(tdf.cols['features'][:], tdf.cols['array_features'][:])

def test_disassemble(sdf):
    assem = VectorAssembler(inputCols=['Pclass', 'Fare', 'Age'], outputCol='features')
    tdf = assem.transform(sdf.dropna())
    tdf = disassemble(tdf, 'features')

    npt.assert_array_equal(tdf.cols['Pclass'][:], tdf.cols['features_0'][:])
    npt.assert_array_equal(tdf.cols['Fare'][:], tdf.cols['features_1'][:])
    npt.assert_array_equal(tdf.cols['Age'][:], tdf.cols['features_2'][:])
