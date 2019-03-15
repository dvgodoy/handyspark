import numpy as np
import numpy.testing as npt
from handyspark import *
import pandas as pd
from pyspark.sql import DataFrame, functions as F
from sklearn.preprocessing import Imputer, KBinsDiscretizer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mutual_info_score
from scipy.spatial import distance
from scipy import stats

def test_to_from_handy(sdf):
    hdf = sdf.toHandy()
    sdf = hdf.notHandy()
    npt.assert_equal(type(hdf), HandyFrame)
    npt.assert_equal(type(sdf), DataFrame)

def test_shape(sdf):
    npt.assert_equal(sdf.toHandy().shape, (891, 12))

def test_response(sdf):
    hdf = sdf.toHandy()
    hdf = hdf.set_response('Survived')
    npt.assert_equal(hdf.is_classification, True)
    npt.assert_equal(hdf.nclasses, 2)
    npt.assert_array_equal(hdf.classes, [0, 1])
    npt.assert_equal(hdf.response, 'Survived')

def test_safety_limit(sdf):
    hdf = sdf.toHandy()
    # maximum 10 elements returned
    hdf.set_safety_limit(10)
    res = hdf.collect()
    npt.assert_equal(len(res), 10)
    npt.assert_equal(hdf._safety, True)
    # deliberately turn safety off -> get everything
    res = hdf.safety_off().collect()
    npt.assert_equal(hdf._safety, True)
    npt.assert_equal(len(res), 891)
    # safety should kick back in
    res = hdf.collect()
    npt.assert_equal(len(res), 10)
    # safety limit does not affect TAKE
    npt.assert_equal(len(hdf.take(20)), 20)
    npt.assert_equal(hdf._safety_limit, 10)

def test_safety_limit2(sdf):
    hdf = sdf.toHandy()
    # maximum 10 elements returned
    hdf.set_safety_limit(10)
    res = hdf.cols[:][:]
    npt.assert_equal(len(res), 10)
    npt.assert_equal(hdf._safety, True)
    # deliberately turn safety off -> get everything
    res = hdf.safety_off().cols[:][:]
    npt.assert_equal(hdf._safety, True)
    npt.assert_equal(len(res), 891)
    # safety should kick back in
    res = hdf.cols[:][:]
    npt.assert_equal(len(res), 10)

def test_values(sdf, pdf):
    hdf = sdf.toHandy()
    hvalues = hdf.limit(10).values
    values = pdf[:10].replace(to_replace=[np.nan], value=[None]).values
    npt.assert_array_equal(hvalues, values)

def test_stages(sdf):
    hdf = sdf.toHandy()
    npt.assert_equal(hdf.stages, 1)
    npt.assert_equal(hdf.groupby('Pclass').agg(F.sum('Fare')).stages, 2)
    npt.assert_equal(hdf.repartition(2).groupby('Pclass').agg(F.sum('Fare')).stages, 3)

def test_value_counts(sdf, pdf):
    hdf = sdf.toHandy()
    hcounts = hdf.cols['Embarked'].value_counts(dropna=True)
    counts = pdf['Embarked'].value_counts().sort_index()
    npt.assert_array_equal(hcounts, counts)

def test_column_values(sdf, pdf):
    hdf = sdf.toHandy()
    npt.assert_array_equal(hdf.cols['Fare'][:20], pdf['Fare'][:20])
    npt.assert_array_equal(hdf.cols['Fare'][:10], pdf['Fare'][:10])

def test_dataframe_values(sdf, pdf):
    hdf = sdf.toHandy()
    npt.assert_array_equal(hdf.cols[['Fare', 'Age']][:20], pdf[['Fare', 'Age']][:20])
    npt.assert_array_equal(hdf.cols[['Fare', 'Age']][:10], pdf[['Fare', 'Age']][:10])

def test_isnull(sdf, pdf):
    hdf = sdf.toHandy()
    hmissing = hdf.isnull()
    hratio = hdf.isnull(ratio=True)
    missing = pdf.isnull().sum()
    ratio = missing / 891.
    npt.assert_array_equal(hmissing, missing)
    npt.assert_array_almost_equal(hratio, ratio)

def test_nunique(sdf, pdf):
    hdf = sdf.toHandy()
    hnunique = hdf.nunique()
    nunique = pdf.nunique()
    approx_error = np.array([-1, 0, 0, 59, 0, -2, 0, 0, 9, -12, 2, 0])
    npt.assert_array_equal(hnunique, nunique + approx_error)

def test_columns_nunique(sdf, pdf):
    hdf = sdf.toHandy()
    hnunique = hdf.cols[['Pclass', 'Embarked']].nunique().squeeze()
    nunique = pdf[['Pclass', 'Embarked']].nunique()
    npt.assert_array_equal(hnunique, nunique)

def test_outliers(sdf, pdf):
    hdf = sdf.toHandy()
    houtliers = hdf.outliers(ratio=True)

    outliers = []
    for colname in hdf.cols.numerical:
        #q1, q3 = hdf._get_summary(colname, '25%')[0], hdf._get_summary(colname, '75%')[0]
        q1, q3 = hdf.cols[colname].q1()[0], hdf.cols[colname].q3()[0]
        iqr = q3 - q1
        lfence = q1 - (1.5 * iqr)
        ufence = q3 + (1.5 * iqr)
        outliers.append((~pdf[colname].dropna().between(lfence, ufence)).sum())
    outliers = pd.Series(outliers, hdf.cols.numerical) / 891.
    npt.assert_array_almost_equal(houtliers, outliers)

def test_mean(sdf, pdf):
    hdf = sdf.toHandy()
    hmean = hdf.cols['continuous'].mean()
    mean = pdf[hdf.cols.continuous].mean()
    npt.assert_array_almost_equal(hmean, mean)

def test_stratified_mean(sdf, pdf):
    hdf = sdf.toHandy()
    hmean = hdf.stratify(['Pclass']).cols['continuous'].mean()
    mean = pdf.groupby(['Pclass'])[hdf.cols.continuous].mean()
    npt.assert_array_almost_equal(hmean, mean)

def test_mode(sdf, pdf):
    hdf = sdf.toHandy()
    hmode = hdf.cols['Embarked'].mode()
    mode = pdf['Embarked'].mode()
    npt.assert_array_equal(hmode, mode)

    hmode = hdf.cols[['Embarked', 'Pclass']].mode()
    mode = pdf[['Embarked', 'Pclass']].mode()
    npt.assert_array_equal(hmode, mode.iloc[0])

    hmode = hdf.stratify(['Pclass']).cols['Embarked'].mode()
    npt.assert_array_equal(hmode, ['S', 'S', 'S'])

def test_median(sdf, pdf):
    hdf = sdf.toHandy()
    hmedian = hdf.cols['Fare'].median(precision=.0001)
    median = pdf['Fare'].median()
    npt.assert_array_equal(hmedian, median)

    hmedian = hdf.cols[['Fare', 'Pclass']].median(precision=.0001)
    median = pdf[['Fare', 'Pclass']].median()
    npt.assert_array_equal(hmedian, median)

    hmedian = hdf.stratify(['Pclass']).cols['Fare'].median(precision=.0001)
    median = pdf.groupby(['Pclass'])['Fare'].median()
    approx_error = np.array([-.8875, -.25, 0.])
    npt.assert_array_almost_equal(hmedian, median + approx_error, decimal=4)

def test_types(sdf):
    hdf = sdf.toHandy()
    hdf2 = hdf.withColumn('newcol', F.lit(1.0))
    npt.assert_array_equal(['PassengerId', 'Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare'], hdf.cols.numerical)
    npt.assert_array_equal(['Age', 'Fare'], hdf.cols.continuous)
    npt.assert_array_equal(['Age', 'Fare', 'newcol'], hdf2.cols.continuous)
    npt.assert_array_equal(['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'SibSp', 'Parch', 'Ticket', 'Cabin',
                            'Embarked'], hdf.cols.categorical)

def test_fill_categorical(sdf):
    hdf = sdf.toHandy()
    hdf_filled = hdf.fill(categorical=['Embarked'])
    hcounts = hdf_filled.cols['Embarked'].value_counts().loc['S']
    npt.assert_equal(hcounts, 646)

def test_fill_continuous(sdf, pdf):
    hdf = sdf.toHandy()
    hdf_filled = hdf.fill(continuous=['Age'], strategy='mean')
    hage = hdf_filled.cols['Age'][:].values

    imputer = Imputer(strategy='mean').fit(pdf[['Age']])
    pdf_filled = imputer.transform(pdf[['Age']])
    age = pdf_filled.ravel()

    npt.assert_array_equal(hage, age)
    npt.assert_array_equal(hdf_filled.statistics_['Age'], imputer.statistics_[0])

def test_sequential_fill(sdf):
    hdf = sdf.toHandy()
    hdf_filled = hdf.stratify(['Pclass']).fill(continuous=['Age'])
    hdf_filled = hdf_filled.fill(categorical=['Embarked'])
    npt.assert_array_equal(sorted(hdf_filled.statistics_.keys()), ['Age', 'Embarked'])
    npt.assert_array_equal(sorted(hdf_filled.statistics_['Age'].keys()),
                           ['Pclass == "1"', 'Pclass == "2"', 'Pclass == "3"'])

def test_corr(sdf, pdf):
    hdf = sdf.toHandy()
    hcorr = hdf.cols[['Fare', 'Age']].corr()
    corr = pdf[['Fare', 'Age']].corr()
    npt.assert_array_almost_equal(hcorr, corr)

def test_stratified_corr(sdf, pdf):
    hdf = sdf.toHandy()
    hcorr = hdf.dropna().stratify(['Pclass']).cols[:].corr()
    corr = pdf.dropna()[sorted(pdf.columns)].groupby(['Pclass']).corr()
    npt.assert_array_almost_equal(hcorr, corr)

def test_fence(sdf, pdf):
    hdf = sdf.toHandy()
    q1, q3 = hdf.approxQuantile(col='Fare', probabilities=[.25, .75], relativeError=0.01)
    hdf_fenced = hdf.fence('Fare')

    fare = pdf['Fare']
    iqr = q3 - q1
    lfence, ufence = q1 - (1.5 * iqr), q3 + (1.5 * iqr)
    fare = fare.mask(fare > ufence, ufence).mask(fare < lfence, lfence)

    npt.assert_array_almost_equal(hdf_fenced.cols['Fare'][:], fare)
    npt.assert_equal(hdf_fenced.fences_['Fare'], [lfence, ufence])

def test_stratified_fence(sdf):
    hdf = sdf.toHandy()
    hdf_fenced = hdf.stratify(['Sex']).fence('Age')

    npt.assert_equal(hdf_fenced.fences_['Age'], {'Sex == "female"': [-9.0, 63.0],
                                                 'Sex == "male"': [-6.0, 66.0]})

def test_grouped_column_values(sdf, pdf):
    hdf = sdf.toHandy()
    hmean = hdf.groupby('Pclass').agg(F.mean('Age').alias('Age')).cols['Age'][:]
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
    hdf = sdf.toHandy()
    sfare = hdf.stratify(['Pclass']).cols['Fare'].mode()
    pfare = pdf.groupby('Pclass').agg({'Fare': lambda v: stats.mode(v)[0]})['Fare']
    npt.assert_array_almost_equal(sfare, pfare)

def test_stratify_list(sdf, pdf):
    # list
    hdf = sdf.toHandy()
    sname = hdf.stratify(['Pclass']).take(1)
    sname = np.array(list(map(lambda row: row.Name, sname)), dtype=np.object)
    pname = pdf.groupby('Pclass')['Name'].first()
    npt.assert_equal(sname, pname)

def test_stratify_pandas_df(sdf, pdf):
    # pd.DataFrame
    hdf = sdf.toHandy()
    scorr = hdf.stratify(['Pclass']).cols[['Fare', 'Age']].corr()
    pcorr = pdf.groupby('Pclass')[['Fare', 'Age']].corr()
    npt.assert_array_almost_equal(scorr.values, pcorr.values)

def test_stratify_pandas_series(sdf, pdf):
    # pd.col
    hdf = sdf.toHandy()
    scounts = hdf.stratify(['Pclass']).cols['Embarked'].value_counts(dropna=True).sort_index()
    pcounts = pdf.groupby('Pclass')['Embarked'].value_counts().sort_index()
    npt.assert_array_almost_equal(scounts, pcounts)

def test_stratify_spark_df(sdf, pdf):
    # pd.col
    hdf = sdf.toHandy()
    sfirst = hdf.dropna().stratify(['Pclass']).limit(1).drop('Pclass').toPandas()
    pfirst = pdf.dropna().groupby('Pclass').first().reset_index(drop=True)
    npt.assert_array_equal(sfirst, pfirst)

def test_stratify_fill(sdf, pdf):
    hdf = sdf.toHandy()
    hdf_filled = hdf.stratify(['Pclass']).fill(continuous=['Age'], categorical=['Embarked'])
    hage = hdf_filled.orderBy('Pclass').cols['Age'][:].values
    hembarked = hdf_filled.orderBy('PassengerId').cols['Embarked'][:].values

    pdf_filled = []
    statistics = {'Age': {}}
    for pclass in [1, 2, 3]:
        filtered = pdf.query('Pclass == {}'.format(pclass))[['Age']]
        imputer = Imputer(strategy='mean').fit(filtered)
        pdf_filled.append(imputer.transform(filtered))
        statistics['Age'].update({'Pclass == "{}"'.format(pclass): imputer.statistics_[0]})
    pdf_filled = np.concatenate(pdf_filled, axis=0)
    age = pdf_filled.ravel()

    npt.assert_array_equal(hage, age)
    npt.assert_array_equal(hembarked, pdf.fillna({'Embarked': 'S'}).sort_values(by='PassengerId')['Embarked'].values)
    npt.assert_array_equal(sorted(list(hdf_filled.statistics_['Age'])),
                           sorted(list(statistics['Age'])))

def test_repr(sdf):
    hdf = sdf.toHandy()
    repr = str(hdf.cols['Fare'])
    npt.assert_equal(repr, "HandyColumns[Fare]")

def test_stratify_bucket(sdf):
    hdf = sdf.toHandy()
    hres = hdf.stratify(['Pclass', Bucket('Age', 3)]).cols['Embarked'].mode()
    npt.assert_equal(hres.values.ravel(), np.array(['S'] * 9))

    hdf = sdf.toHandy()
    hres = hdf.stratify(['Pclass', Bucket('Age', 3)]).cols['Embarked'].value_counts().sort_index()
    npt.assert_equal(hres.values.ravel(), np.array([21, 23, 40, 2, 68, 13, 17, 8, 59, 7, 1, 86,
                                                    1, 11, 28, 14, 166, 13, 8, 119, 2, 5]))

def test_stratified_nunique(sdf, pdf):
    hdf = sdf.toHandy()
    hnunique = hdf.stratify(['Pclass']).cols['Cabin'].nunique()
    nunique = pdf.groupby(['Pclass'])['Cabin'].nunique()
    npt.assert_array_equal(hnunique, nunique)

def test_mahalanobis(sdf, pdf):
    colnames = ['Fare', 'SibSp', 'Parch']
    hdf = sdf.toHandy()
    hres = hdf._handy._calc_mahalanobis_distance(colnames).toHandy().cols['__mahalanobis'][:].values
    pipeline = make_pipeline(StandardScaler())
    pdf = pd.DataFrame(pipeline.fit_transform(pdf[colnames]), columns=colnames)
    invmat = np.linalg.inv(pdf.cov())
    res = pdf.apply(lambda row: distance.mahalanobis(row.values, np.zeros_like(row.values), invmat), axis=1)
    npt.assert_array_almost_equal(hres, res, decimal=4)

def test_entropy(sdf, pdf):
    hdf = sdf.toHandy()
    hres = hdf.cols['Pclass'].entropy()
    res = stats.entropy(pdf.groupby('Pclass').count().iloc[:, 0], base=2)
    npt.assert_array_almost_equal(hres, res)

def test_mutual_info(sdf, pdf):
    hdf = sdf.toHandy()
    hres = hdf.cols[['Survived', 'Pclass']].mutual_info()
    res = mutual_info_score(pdf['Survived'], pdf['Pclass'])
    # converts to log2
    res = np.log2(np.exp(res))
    npt.assert_array_almost_equal(hres.loc['Survived', 'Pclass'], res)
