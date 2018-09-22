import numpy as np
import pandas as pd
from handyspark.util import dense_to_array, disassemble
from operator import add
from pyspark.ml.stat import Correlation
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.pipeline import Pipeline
from pyspark.mllib.common import _java2py, _py2java
from pyspark.mllib.stat.test import KolmogorovSmirnovTestResult
from pyspark.sql import Row, functions as F
from scipy.linalg import inv
from scipy.stats import chi2

def mahalanobis(sdf, colnames):
    assembler = VectorAssembler(inputCols=colnames, outputCol='__features')
    scaler = StandardScaler(inputCol='__features', outputCol='__scaled', withMean=True)
    pipeline = Pipeline(stages=[assembler, scaler])
    features = pipeline.fit(sdf).transform(sdf)

    mat = Correlation.corr(features, '__scaled').head()[0].toArray()
    inv_mat = inv(mat)
    critical_value = chi2.ppf(0.999, len(colnames))

    @F.pandas_udf('double')
    def pudf_mult(v):
        return v.apply(lambda v: np.dot(np.dot(np.transpose(v), inv_mat), v))

    features = dense_to_array(features, '__scaled', '__array_scaled')
    distance = (features
                .withColumn('__mahalanobis', pudf_mult('__array_scaled'))
                .withColumn('__outlier', F.col('__mahalanobis') > critical_value)
                .drop('__features', '__scaled', '__array_scaled'))
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
    res = np.zeros(shape=(n, n))
    for i in range(n):
        for j in range(i, n):
            tdf = VectorAssembler(inputCols=[colnames[i], colnames[j]], outputCol='__vectors').transform(sdf)
            tdf = probabilities(tdf, '__vectors')
            tdf = disassemble(dense_to_array(tdf, '__col', '__features'), '__features')
            tdf = tdf.join(probs[i].toDF('__features_0', '__p0'), on='__features_0')
            tdf = tdf.join(probs[j].toDF('__features_1', '__p1'), on='__features_1')
            mi = tdf.select(F.sum(F.expr('log2(__probability / (__p0 * __p1)) * __probability'))).take(1)[0][0]
            res[i, j] = mi
            res[j, i] = mi
    return pd.DataFrame(res, index=colnames, columns=colnames)

def StatisticalSummaryValues(sdf, colnames):
    if not isinstance(colnames, (list, tuple)):
        colnames = [colnames]
    jvm = sdf._sc._jvm
    summ = sdf.select(colnames).describe().toPandas().set_index('summary')
    ssvs = {}
    for colname in colnames:
        values = list(map(float, summ[colname].values))
        values = values[1], np.sqrt(values[2]), int(values[0]), values[4], values[3], values[0] * values[1]
        java_class = jvm.org.apache.commons.math3.stat.descriptive.StatisticalSummaryValues
        ssvs.update({colname: java_class(*values)})
    return ssvs

def tTest(jvm, *ssvs):
    n = len(ssvs)
    res = np.identity(n)
    java_class = jvm.org.apache.commons.math3.stat.inference.TTest
    java_obj = java_class()
    for i in range(n):
        for j in range(i + 1, n):
            pvalue = java_obj.tTest(ssvs[i], ssvs[j])
            res[i, j] = pvalue
            res[j, i] = pvalue
    return res

def KolmogorovSmirnovTest(sdf, colname, dist='normal', *params):
    _distributions = ['Beta', 'Cauchy', 'ChiSquared', 'Exponential', ' F', 'Gamma', 'Gumbel', 'Laplace', 'Levy',
                      'Logistic', 'LogNormal', 'Nakagami', 'Normal', 'Pareto', 'T', 'Triangular', 'Uniform', 'Weibull']
    _distlower = list(map(lambda v: v.lower(), _distributions))
    try:
        dist = _distributions[_distlower.index(dist)]
        if dist == 'Uniform':
            dist += 'Real'
    except ValueError:
        dist = 'Normal'
        params = (0., 1.)
    jvm = sdf._sc._jvm
    rdd = sdf.select(colname).rdd.map(lambda t: t[0])
    ks = jvm.org.apache.spark.mllib.stat.test.KolmogorovSmirnovTest
    java_class = getattr(jvm, 'org.apache.commons.math3.distribution.{}Distribution'.format(dist))
    java_obj = java_class(*params)
    jrdd = _py2java(sdf._sc, rdd)
    res = ks.testOneSample(jrdd.rdd(), java_obj)
    return KolmogorovSmirnovTestResult(res)
