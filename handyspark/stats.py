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

g = jvm.com.google.common.primitives.Doubles
go = g.toArray([0., 1., 2.])
go2 = g.toArray([5., 6., 7.])

java_class = jvm.org.apache.commons.math3.stat.inference.TTest
jo = java_class()
jo.tTest(go, go2)

ssv = jvm.org.apache.commons.math3.stat.descriptive.StatisticalSummaryValues
ssvo = ssv(0., 1., 100, 1., -1., 0.)
ssvo2 = ssv(0.5, 1.5, 100, 2., -1., 50.)
jo.tTest(ssvo, ssvo2)

ks = jvm.org.apache.commons.math3.stat.inference.KolmogorovSmirnovTest
nd = jvm.org.apache.commons.math3.distribution.NormalDistribution
jdata = g.toArray(np.random.randn(100))
ks().kolmogorovSmirnovTest(nd(0., 1.), jdata)