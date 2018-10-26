import numpy as np
import pandas as pd
from handyspark.util import dense_to_array, disassemble, check_columns, ensure_list
from operator import add
from pyspark.ml.stat import Correlation
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.pipeline import Pipeline
from pyspark.mllib.common import _py2java
from pyspark.mllib.stat.test import KolmogorovSmirnovTestResult
from pyspark.sql import Row, functions as F
from scipy.linalg import inv
from scipy.stats import chi2

def mahalanobis(sdf, colnames):
    """Computes Mahalanobis distance from origin and compares to critical values
    using Chi-Squared distribution to identify possible outliers.
    """
    check_columns(sdf, colnames)
    # Builds pipeline to assemble feature columns and scale them
    assembler = VectorAssembler(inputCols=colnames, outputCol='__features')
    scaler = StandardScaler(inputCol='__features', outputCol='__scaled', withMean=True)
    pipeline = Pipeline(stages=[assembler, scaler])
    features = pipeline.fit(sdf).transform(sdf)

    # Computes correlation between features and inverts it
    # Since we scaled the features, we can assume they have unit variance
    # and therefore, correlation and covariance matrices are the same!
    mat = Correlation.corr(features, '__scaled').head()[0].toArray()
    inv_mat = inv(mat)

    # Computes critical value
    critical_value = chi2.ppf(0.999, len(colnames))

    # Builds Pandas UDF to compute Mahalanobis distance from origin
    # sqrt((V - 0) * inv_M * (V - 0))
    try:
        import pyarrow
        @F.pandas_udf('double')
        def pudf_mult(v):
            return v.apply(lambda v: np.sqrt(np.dot(np.dot(v, inv_mat), v)))
    except:
        @F.udf('double')
        def pudf_mult(v):
            return v.apply(lambda v: np.sqrt(np.dot(np.dot(v, inv_mat), v)))

    # Convert feature vector into array
    features = dense_to_array(features, '__scaled', '__array_scaled')
    # Computes Mahalanobis distance and flags as outliers all elements above critical value
    distance = (features
                .withColumn('__mahalanobis', pudf_mult('__array_scaled'))
                .withColumn('__outlier', F.col('__mahalanobis') > critical_value)
                .drop('__features', '__scaled', '__array_scaled'))
    return distance

def distribution(sdf, colname):
    check_columns(sdf, colname)
    rdd = sdf.notHandy().select(colname).rdd.map(lambda row: (row[0], 1))
    n = rdd.count()
    return rdd.reduceByKey(add).map(lambda t: Row(__col=t[0], __probability=t[1]/n)).toDF()

def entropy(sdf, colnames):
    colnames = ensure_list(colnames)
    check_columns(sdf, colnames)
    entropy = []
    for colname in colnames:
        entropy.append(distribution(sdf, colname)
                       .select(F.sum(F.expr('-log2(__probability)*__probability'))).take(1)[0][0])
    return pd.Series(entropy, index=colnames)

def mutual_info(sdf, colnames):
    check_columns(sdf, colnames)
    n = len(colnames)
    probs = []
    for i in range(n):
        probs.append(distribution(sdf, colnames[i]))
    res = np.zeros(shape=(n, n))
    for i in range(n):
        for j in range(i, n):
            tdf = VectorAssembler(inputCols=[colnames[i], colnames[j]], outputCol='__vectors').transform(sdf)
            tdf = distribution(tdf, '__vectors')
            tdf = disassemble(dense_to_array(tdf, '__col', '__features'), '__features')
            tdf = tdf.join(probs[i].toDF('__features_0', '__p0'), on='__features_0')
            tdf = tdf.join(probs[j].toDF('__features_1', '__p1'), on='__features_1')
            mi = tdf.select(F.sum(F.expr('log2(__probability / (__p0 * __p1)) * __probability'))).take(1)[0][0]
            res[i, j] = mi
            res[j, i] = mi
    return pd.DataFrame(res, index=colnames, columns=colnames)

def StatisticalSummaryValues(sdf, colnames):
    """Builds a Java StatisticalSummaryValues object for each column
    """
    colnames = ensure_list(colnames)
    check_columns(sdf, colnames)

    jvm = sdf._sc._jvm
    summ = sdf.notHandy().select(colnames).describe().toPandas().set_index('summary')
    ssvs = {}
    for colname in colnames:
        values = list(map(float, summ[colname].values))
        values = values[1], np.sqrt(values[2]), int(values[0]), values[4], values[3], values[0] * values[1]
        java_class = jvm.org.apache.commons.math3.stat.descriptive.StatisticalSummaryValues
        ssvs.update({colname: java_class(*values)})
    return ssvs

def tTest(jvm, *ssvs):
    """Performs a t-Test for difference of means using StatisticalSummaryValues objects
    """
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
    """Performs a KolmogorovSmirnov test for comparing the distribution of values in a column
    to a named canonical distribution.
    """
    check_columns(sdf, colname)
    # Supported distributions
    _distributions = ['Beta', 'Cauchy', 'ChiSquared', 'Exponential', ' F', 'Gamma', 'Gumbel', 'Laplace', 'Levy',
                      'Logistic', 'LogNormal', 'Nakagami', 'Normal', 'Pareto', 'T', 'Triangular', 'Uniform', 'Weibull']
    _distlower = list(map(lambda v: v.lower(), _distributions))
    try:
        dist = _distributions[_distlower.index(dist)]
        # the actual name for the Uniform distribution is UniformReal
        if dist == 'Uniform':
            dist += 'Real'
    except ValueError:
        # If we cannot find a distribution, fall back to Normal
        dist = 'Normal'
        params = (0., 1.)
    jvm = sdf._sc._jvm
    # Maps the DF column into a numeric RDD and turns it into Java RDD
    rdd = sdf.notHandy().select(colname).rdd.map(lambda t: t[0])
    jrdd = _py2java(sdf._sc, rdd)
    # Gets the Java class of the corresponding distribution and creates an obj
    java_class = getattr(jvm, 'org.apache.commons.math3.distribution.{}Distribution'.format(dist))
    java_obj = java_class(*params)
    # Loads the KS test class and performs the test
    ks = jvm.org.apache.spark.mllib.stat.test.KolmogorovSmirnovTest
    res = ks.testOneSample(jrdd.rdd(), java_obj)
    return KolmogorovSmirnovTestResult(res)
