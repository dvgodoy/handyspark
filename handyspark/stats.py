import numpy as np
from handyspark.util import dense_to_array, disassemble
from operator import add
from pyspark.ml.stat import Correlation
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.pipeline import Pipeline
from pyspark.sql import Row, functions as F
from scipy.linalg import inv

def mahalanobis(sdf, colnames):
    assembler = VectorAssembler(inputCols=colnames, outputCol='__features')
    scaler = StandardScaler(inputCol='__features', outputCol='__scaled', withMean=True)
    pipeline = Pipeline(stages=[assembler, scaler])
    features = pipeline.fit(sdf).transform(sdf)

    mat = Correlation.corr(features, '__scaled').head()[0].toArray()

    @F.udf('double')
    def udf_mult(v):
        return float(np.dot(np.dot(np.transpose(v), inv(mat)), v))

    distance = features.withColumn('__mahalanobis', udf_mult('__scaled')).drop('__features', '__scaled')
    return distance

def add_probabilities(sdf, colname, prob):
    rdd = sdf.select(colname, prob).rdd.map(list)
    return rdd.reduceByKey(add).map(lambda t: Row(*t)).toDF([colname, prob])

def probabilities(sdf, colname):
    rdd = sdf.select(colname).rdd.map(lambda row: (row[0], 1))
    n = rdd.count()
    return rdd.reduceByKey(add).map(lambda t: Row(col=t[0], __probability=t[1]/n)).toDF()

def entropy(sdf, colname):
    return probabilities(sdf, colname).select(F.sum(F.expr('-log2(__probability)*__probability'))).take(1)[0][0]

def mutual_info(sdf, col1, col2):
    tdf = VectorAssembler(inputCols=[col1, col2], outputCol='__vectors').transform(sdf)
    tdf = probabilities(tdf, '__vectors')
    tdf = disassemble(dense_to_array(tdf, '__col', '__features'), '__features')
    p0 = add_probabilities(tdf, '__features_0', '__probability')
    p1 = add_probabilities(tdf, '__features_1', '__probability')
    tdf = (tdf
          .join(p0.withColumnRenamed('__probability', '__p0'), on='__features_0')
          .join(p1.withColumnRenamed('__probability', '__p1'), on='__features_1'))
    return (tdf.withColumn('__mi',
                           F.expr('log2(__probability / (__p0 * __p1)) * __probability')).select(F.sum('__mi'))
            .take(1)[0][0])

# g = jvm.com.google.common.primitives.Doubles
# go = g.toArray([0., 1., 2.])
# go2 = g.toArray([5., 6., 7.])
#
# java_class = jvm.org.apache.commons.math3.stat.inference.TTest
# jo = java_class()
# jo.tTest(go, go2)
#
# ssv = jvm.org.apache.commons.math3.stat.descriptive.StatisticalSummaryValues
# ssvo = ssv(0., 1., 100, 1., -1., 0.)
# ssvo2 = ssv(0.5, 1.5, 100, 2., -1., 50.)
# jo.tTest(ssvo, ssvo2)
#
# ks = jvm.org.apache.commons.math3.stat.inference.KolmogorovSmirnovTest
# nd = jvm.org.apache.commons.math3.distribution.NormalDistribution
# jdata = g.toArray(np.random.randn(100))
# ks().kolmogorovSmirnovTest(nd(0., 1.), jdata)
