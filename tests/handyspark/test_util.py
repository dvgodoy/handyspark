import numpy.testing as npt
from pyspark.ml.feature import VectorAssembler
from handyspark.util import dense_to_array, disassemble

def test_dense_to_array(sdf):
    assem = VectorAssembler(inputCols=['Pclass', 'Fare', 'Age'], outputCol='features')
    tdf = assem.transform(sdf.dropna())
    tdf = dense_to_array(tdf, 'features', 'array_features')

    npt.assert_array_equal(tdf.handy['features', None], tdf.col['array_features', None])

def test_disassemble(sdf):
    assem = VectorAssembler(inputCols=['Pclass', 'Fare', 'Age'], outputCol='features')
    tdf = assem.transform(sdf.dropna())
    tdf = disassemble(tdf, 'features')

    npt.assert_array_equal(tdf.handy['Pclass', None], tdf.col['features_0', None])
    npt.assert_array_equal(tdf.handy['Fare', None], tdf.col['features_1', None])
    npt.assert_array_equal(tdf.handy['Age', None], tdf.col['features_2', None])
