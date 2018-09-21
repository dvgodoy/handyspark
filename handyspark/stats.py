import numpy as np
import pandas as pd
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

def probabilities(sdf, colname):
    rdd = sdf.select(colname).rdd.map(lambda row: (row[0], 1))
    n = rdd.count()
    return rdd.reduceByKey(add).map(lambda t: Row(__col=t[0], __probability=t[1]/n)).toDF()

def entropy(sdf, colnames):
    if not isinstance(colnames, (list, tuple)):
        colnames = [colnames]
    entropy = []
    for colname in colnames:
        entropy.append(probabilities(sdf, colname)
                       .select(F.sum(F.expr('-log2(__probability)*__probability'))).take(1)[0][0])
    return pd.Series(entropy, index=colnames)

def mutual_info(sdf, colnames):
    n = len(colnames)
    probs = []
    for i in range(n):
        probs.append(probabilities(sdf, colnames[i]))
    res = np.identity(n)
    for i in range(n):
        for j in range(i + 1, n):
            tdf = VectorAssembler(inputCols=[colnames[i], colnames[j]], outputCol='__vectors').transform(sdf)
            tdf = probabilities(tdf, '__vectors')
            tdf = disassemble(dense_to_array(tdf, '__col', '__features'), '__features')
            tdf = tdf.join(probs[i].toDF('__features_0', '__p0'), on='__features_0')
            tdf = tdf.join(probs[j].toDF('__features_1', '__p1'), on='__features_1')
            mi = tdf.select(F.sum(F.expr('log2(__probability / (__p0 * __p1)) * __probability'))).take(1)[0][0]
            res[i, j] = mi
            res[j, i] = mi
    return pd.DataFrame(res, index=colnames, columns=colnames)

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
