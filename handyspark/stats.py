import numpy as np
from pyspark.ml.stat import Correlation
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.pipeline import Pipeline
from pyspark.sql.types import DoubleType
from pyspark.sql.udf import UserDefinedFunction
from scipy.linalg import inv

def mahalanobis(sdf, colnames):
    assembler = VectorAssembler(inputCols=colnames, outputCol='features')
    scaler = StandardScaler(inputCol='features', outputCol='scaled', withMean=True)
    pipeline = Pipeline(stages=[assembler, scaler])
    features = pipeline.fit(sdf).transform(sdf)

    mat = Correlation.corr(features, 'scaled').head()[0].toArray()

    def mult(v):
        return float(np.dot(np.dot(np.transpose(v), inv(mat)), v))

    udf_mult = UserDefinedFunction(mult, DoubleType())

    distance = features.select(udf_mult('scaled').alias('mahalanobis'))
    return distance