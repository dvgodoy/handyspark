import numpy as np
from pyspark.ml.stat import Correlation
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.pipeline import Pipeline
from pyspark.sql.types import DoubleType
from pyspark.sql.functions import udf
from scipy.linalg import inv

def mahalanobis(sdf, colnames):
    assembler = VectorAssembler(inputCols=colnames, outputCol='__features')
    scaler = StandardScaler(inputCol='__features', outputCol='__scaled', withMean=True)
    pipeline = Pipeline(stages=[assembler, scaler])
    features = pipeline.fit(sdf).transform(sdf)

    mat = Correlation.corr(features, '__scaled').head()[0].toArray()

    @udf('double')
    def udf_mult(v):
        return float(np.dot(np.dot(np.transpose(v), inv(mat)), v))

    distance = features.withColumn('__mahalanobis', udf_mult('__scaled')).drop('__features', '__scaled')
    return distance