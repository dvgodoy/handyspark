import numpy as np
from handyspark.util import check_columns, ensure_list
from pyspark.mllib.common import _py2java
from pyspark.mllib.stat.test import KolmogorovSmirnovTestResult

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
