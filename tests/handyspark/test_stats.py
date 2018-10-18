import numpy as np
import numpy.testing as npt
import pandas as pd
import handyspark
from handyspark.stats import mahalanobis, entropy, mutual_info, KolmogorovSmirnovTest
from pyspark.sql import functions as F
from scipy.spatial import distance
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mutual_info_score

def test_mahalanobis(sdf, pdf):
    colnames = ['Fare', 'Pclass', 'SibSp', 'Parch']
    hdf = sdf.toHandy()
    hres = mahalanobis(hdf, colnames).toHandy().cols['__mahalanobis'][:].values
    pdf = pd.DataFrame(StandardScaler().fit_transform(pdf[colnames]), columns=colnames)
    invmat = np.linalg.inv(pdf.corr())
    res = pdf.apply(lambda row: distance.mahalanobis(row.values, np.zeros_like(row.values), invmat), axis=1)
    npt.assert_array_almost_equal(hres, res, decimal=2)

def test_entropy(sdf, pdf):
    hdf = sdf.toHandy()
    hres = entropy(hdf, 'Pclass')
    res = stats.entropy(pdf.groupby('Pclass').count().iloc[:, 0], base=2)
    npt.assert_array_almost_equal(hres, res)

def test_mutual_info(sdf, pdf):
    hdf = sdf.toHandy()
    hres = mutual_info(hdf, ['Survived', 'Pclass']).iloc[0, 1]
    res = mutual_info_score(pdf['Survived'], pdf['Pclass'])
    # converts to log2
    res = np.log2(np.exp(res))
    npt.assert_array_almost_equal(hres, res)

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