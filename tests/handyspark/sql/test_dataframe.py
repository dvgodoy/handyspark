import numpy as np
import numpy.testing as npt
from handyspark import *
from handyspark.sql.dataframe import Bucket, Quantile
from pyspark.sql import DataFrame, functions as F
from scipy.stats import mode
from sklearn.preprocessing import Imputer, KBinsDiscretizer

def test_to_from_handy(sdf):
    hdf = sdf.toHandy
    sdf = hdf.notHandy
    npt.assert_equal(type(hdf), HandyFrame)
    npt.assert_equal(type(sdf), DataFrame)

def test_shape(sdf):
    npt.assert_equal(sdf.toHandy.shape, (891, 12))

def test_response(sdf):
    hdf = sdf.toHandy
    hdf = hdf.set_response('Survived')
    npt.assert_equal(hdf.is_classification, True)
    npt.assert_equal(hdf.nclasses, 2)
    npt.assert_array_equal(hdf.classes, [0, 1])
    npt.assert_equal(hdf.response, 'Survived')

def test_safety_limit(sdf):
    hdf = sdf.toHandy
    # maximum 10 elements returned
    hdf.set_safety_limit(10)
    npt.assert_equal(len(hdf.collect()), 10)
    # deliberately turn safety off -> get everything
    npt.assert_equal(len(hdf.safety_off().collect()), 891)
    # safety limit does not affect TAKE
    npt.assert_equal(len(hdf.take(20)), 20)

def test_values(sdf, pdf):
    hdf = sdf.toHandy
    hvalues = hdf.limit(10).values
    values = pdf[:10].replace(to_replace=[np.nan], value=[None]).values
    npt.assert_array_equal(hvalues, values)

def test_stages(sdf):
    hdf = sdf.toHandy
    npt.assert_equal(hdf.stages, 1)
    npt.assert_equal(hdf.groupby('Pclass').agg(F.sum('Fare')).stages, 2)
    npt.assert_equal(hdf.repartition(2).groupby('Pclass').agg(F.sum('Fare')).stages, 3)

def test_value_counts(sdf, pdf):
    hdf = sdf.toHandy
    hcounts = hdf.value_counts('Embarked', keepna=False)
    counts = pdf['Embarked'].value_counts()
    npt.assert_array_equal(hcounts, counts)

def test_column_values(sdf, pdf):
    hdf = sdf.toHandy
    npt.assert_array_equal(hdf.handy['Fare'], pdf['Fare'][:20])
    npt.assert_array_equal(hdf.handy['Fare', 10], pdf['Fare'][:10])

def test_isnull(sdf, pdf):
    hdf = sdf.toHandy
    hmissing = hdf.isnull()
    hratio = hdf.isnull(ratio=True)
    missing = pdf.isnull().sum()
    ratio = missing / 891.
    npt.assert_array_equal(hmissing, missing)
    npt.assert_array_almost_equal(hratio, ratio)

def test_nunique(sdf, pdf):
    hdf = sdf.toHandy
    hnunique = hdf.nunique()
    nunique = pdf.nunique()
    npt.assert_array_equal(hnunique, nunique)

def test_mode(sdf, pdf):
    hdf = sdf.toHandy
    hmode = hdf.mode('Embarked')
    mode = pdf['Embarked'].mode()[0]
    npt.assert_equal(hmode, mode)

def test_fill_categorical(sdf):
    hdf = sdf.toHandy
    hdf_filled = hdf.fill(categorical=['Embarked'])
    hcounts = hdf_filled.value_counts('Embarked').loc['S']
    npt.assert_equal(hcounts, 646)

def test_fill_continuous(sdf, pdf):
    hdf = sdf.toHandy
    hdf_filled = hdf.fill(continuous=['Age'], strategy='mean')
    hage = hdf_filled.handy['Age', None].values

    imputer = Imputer(strategy='mean').fit(pdf[['Age']])
    pdf_filled = imputer.transform(pdf[['Age']])
    age = pdf_filled.ravel()

    npt.assert_array_equal(hage, age)
    npt.assert_array_equal(hdf_filled.statistics_['Age'], imputer.statistics_[0])

def test_corr(sdf, pdf):
    hdf = sdf.toHandy
    hcorr = hdf.corr_matrix(['Fare', 'Age'])
    corr = pdf[['Fare', 'Age']].corr()
    npt.assert_array_almost_equal(hcorr, corr)

def test_fence(sdf, pdf):
    hdf = sdf.toHandy
    q1, q3 = hdf.approxQuantile(col='Fare', probabilities=[.25, .75], relativeError=0.01)
    hdf_fenced = hdf.fence('Fare')

    fare = pdf['Fare']
    iqr = q3 - q1
    lfence, ufence = q1 - (1.5 * iqr), q3 + (1.5 * iqr)
    fare = fare.mask(fare > ufence, ufence).mask(fare < lfence, lfence)

    npt.assert_array_almost_equal(hdf_fenced.handy['Fare', None], fare)
    npt.assert_equal(hdf_fenced.fences_['Fare'], [lfence, ufence])

def test_grouped_column_values(sdf, pdf):
    hdf = sdf.toHandy
    hmean = hdf.groupby('Pclass').agg(F.mean('Age').alias('Age')).handy['Age']
    mean = pdf.groupby('Pclass').agg({'Age': np.mean})['Age']
    npt.assert_array_equal(hmean, mean)

def test_bucket(sdf, pdf):
    bucket = Bucket('Age', bins=3)
    sbuckets = bucket._get_buckets(sdf.fillna(0.0))[1:-1]

    kbins = KBinsDiscretizer(n_bins=3, strategy='uniform')
    kbins.fit(pdf[['Age']].fillna(0.0))
    pbuckets = kbins.bin_edges_[0]

    npt.assert_almost_equal(sbuckets, pbuckets)

def test_quantile(sdf, pdf):
    bucket = Quantile('Age', bins=3)
    sbuckets = bucket._get_buckets(sdf.fillna(0.0))[1:-1]

    kbins = KBinsDiscretizer(n_bins=3, strategy='quantile')
    kbins.fit(pdf[['Age']].fillna(0.0))
    pbuckets = kbins.bin_edges_[0]

    npt.assert_almost_equal(sbuckets, pbuckets)

def test_stratify_length(sdf, pdf):
    # matches lengths only
    hdf = sdf.toHandy
    sfare = hdf.stratify(['Pclass']).mode('Fare')
    pfare = pdf.groupby('Pclass').agg({'Fare': lambda v: mode(v)[0]})
    npt.assert_array_almost_equal(sfare, pfare)

def test_stratify_list(sdf, pdf):
    # list
    hdf = sdf.toHandy
    sname = hdf.stratify(['Pclass']).take(1)
    sname = np.array(list(map(lambda row: row.Name, sname)), dtype=np.object)
    pname = pdf.groupby('Pclass')['Name'].first()
    npt.assert_equal(sname, pname)

def test_stratify_pandas_df(sdf, pdf):
    # pd.DataFrame
    hdf = sdf.toHandy
    scorr = hdf.stratify(['Pclass']).corr_matrix(['Fare', 'Age'])
    pcorr = pdf.groupby('Pclass')[['Fare', 'Age']].corr()
    npt.assert_array_almost_equal(scorr.values, pcorr.values)

def test_stratify_pandas_series(sdf, pdf):
    # pd.Series
    hdf = sdf.toHandy
    scounts = hdf.stratify(['Pclass']).value_counts('Embarked', keepna=False)
    pcounts = pdf.groupby('Pclass')['Embarked'].value_counts().sort_index()
    npt.assert_array_almost_equal(scounts, pcounts)

def test_stratify_spark_df(sdf, pdf):
    # pd.Series
    hdf = sdf.toHandy
    sfirst = hdf.dropna().stratify(['Pclass']).limit(1).drop('Pclass').toPandas()
    pfirst = pdf.dropna().groupby('Pclass').first().reset_index(drop=True)
    npt.assert_array_equal(sfirst, pfirst)

def test_stratify_fill(sdf, pdf):
    hdf = sdf.toHandy
    hdf_filled = hdf.stratify(['Pclass']).fill(continuous=['Age'])
    hage = hdf_filled.handy['Age', None].values

    pdf_filled = []
    statistics = {}
    for pclass in [1, 2, 3]:
        filtered = pdf.query('Pclass == {}'.format(pclass))[['Age']]
        imputer = Imputer(strategy='mean').fit(filtered)
        pdf_filled.append(imputer.transform(filtered))
        statistics.update({'Pclass == "{}"'.format(pclass): {'Age': imputer.statistics_[0]}})
    pdf_filled = np.concatenate(pdf_filled, axis=0)
    age = pdf_filled.ravel()

    npt.assert_array_equal(hage, age)
    npt.assert_array_equal(sorted(list(hdf_filled.statistics_.items())),
                           sorted(list(statistics.items())))
