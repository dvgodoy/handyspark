import numpy.testing as npt
from handyspark.stats import KolmogorovSmirnovTest
from pyspark.sql import functions as F

def test_ks(sdf):
    # generates uniform
    sdf = sdf.withColumn('rand', F.rand(42))
    # compares with uniform,it should NOT reject
    pval = KolmogorovSmirnovTest(sdf, 'rand', dist='uniform').pValue
    npt.assert_equal(pval > .05, True)
    # compares with normal, it SHOULD reject
    pval = KolmogorovSmirnovTest(sdf, 'rand').pValue
    npt.assert_equal(pval < .05, True)

    # generates normal
    sdf = sdf.withColumn('rand', F.randn(42))
    # compares with normal, it should NOT reject
    pval = KolmogorovSmirnovTest(sdf, 'rand').pValue
    npt.assert_equal(pval > .05, True)
    # compares with uniform, it SHOULD reject
    pval = KolmogorovSmirnovTest(sdf, 'rand', dist='uniform').pValue
    npt.assert_equal(pval < .05, True)