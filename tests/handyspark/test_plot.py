import base64
import numpy.testing as npt
import numpy as np
import seaborn as sns
from handyspark import *
from handyspark.plot import consolidate_plots, strat_histogram
from io import BytesIO
from matplotlib import pyplot as plt

def plot_to_base64(fig):
    bytes_data = BytesIO()
    fig.savefig(bytes_data, format='png')
    bytes_data.seek(0)
    b64_data = base64.b64encode(bytes_data.read())
    plt.close(fig)
    return b64_data

def plot_to_pixels(fig, shape=None):
    fig.canvas.draw()
    rgb_data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    plt.clf()
    plt.cla()
    plt.close(fig)
    if shape is None:
        rgb_data = rgb_data.reshape((int(len(rgb_data) / 3), 3))
    else:
        rgb_data = rgb_data.reshape(shape)
    return rgb_data

def test_boxplot_single(sdf, pdf):
    pax = pdf[['Fare']].boxplot(showfliers=False)
    pax.legend().remove()
    pax.set_ylabel('')
    p64 = plot_to_base64(pax.figure)

    hdf = sdf.toHandy()
    sax = hdf.cols['Fare'].boxplot(showfliers=False, precision=.0001)
    s64 = plot_to_base64(sax.figure)
    npt.assert_equal(p64, s64)

def test_boxplot_multiple(sdf, pdf):
    pax = pdf[['Fare', 'Age']].boxplot(showfliers=False)
    pax.legend().remove()
    pax.set_ylabel('')
    p64 = plot_to_pixels(pax.figure, (480, 640, 3))

    # Spark computes quartiles approximately,
    # so it results in a small difference between the plots
    hdf = sdf.toHandy()
    sax = hdf.cols[['Fare', 'Age']].boxplot(showfliers=False, precision=.0001)
    s64 = plot_to_pixels(sax.figure, (480, 640, 3))

    diff = s64 - p64
    npt.assert_equal(diff.sum(), 110414)
    npt.assert_equal((diff != 0).sum(), 871)

def test_hist_categorical(sdf, pdf):
    hdf = sdf.toHandy()
    sax = hdf.dropna(subset=['Embarked']).cols['Embarked'].hist()
    s64 = plot_to_base64(sax.figure)

    pdf = pdf.groupby(['Embarked'])['PassengerId'].count().sort_index()
    pax = pdf.plot(kind='bar', color='C0', legend=False, rot=0, ax=None, title='Embarked')
    p64 = plot_to_base64(pax.figure)

    npt.assert_equal(p64, s64)

def test_hist_continuous(sdf, pdf):
    hdf = sdf.toHandy()
    sax = hdf.cols['Fare'].hist(bins=5)
    s64 = plot_to_base64(sax.figure)

    pax = pdf[['Fare']].plot.hist(bins=5)
    pax.legend().remove()
    pax.set_ylabel('')
    pax.set_title('Fare')
    p64 = plot_to_base64(pax.figure)

    npt.assert_equal(p64, s64)

def test_scatterplot(sdf, pdf):
    hdf = sdf.toHandy()
    sax = hdf.fillna({'Age': 29.0}).cols[['Fare', 'Age']].scatterplot()
    sax.set_xlim([0, 515])
    sax.set_ylim([0, 85])
    s64 = plot_to_pixels(sax.figure, (480, 640, 3))

    # Traditional plot is not bucketized!
    pdf = pdf.fillna({'Age': 29.0})
    df_counts = pdf.groupby(['Fare', 'Age'])['PassengerId'].count().to_frame('Proportion')
    df_counts.loc[:, 'Proportion'] = df_counts['Proportion'].apply(lambda p: round(p / 891, 4))
    pax = sns.scatterplot(data=df_counts.reset_index(),
                           x='Fare',
                           y='Age',
                           size='Proportion',
                           legend=False)
    pax.set_xlim([0, 515])
    pax.set_ylim([0, 85])
    p64 = plot_to_pixels(pax.figure, (480, 640, 3))

    # Differences arise from bucketized vs not bucketized scatterplots
    diff = s64 - p64
    npt.assert_equal(diff.sum(), 4759745)
    npt.assert_equal((diff != 0).sum(), 45616)

def test_stratified_boxplot(sdf, pdf):
    hdf = sdf.toHandy()
    sfig, _ = hdf.stratify(['Pclass']).cols['Fare'].boxplot(showfliers=False, precision=.0001)
    s64 = plot_to_pixels(sfig, (480, 640, 3))

    pax = pdf.boxplot('Fare', by='Pclass', showfliers=False)
    pax.set_xlabel('')
    plt.suptitle('')
    plt.xticks([1, 2, 3], ['Pclass={}'.format(i) for i in [1, 2, 3]])
    plt.tight_layout()
    p64 = plot_to_pixels(pax.figure, (480, 640, 3))

    # Differences arise from quantile calculations
    diff = s64 - p64
    npt.assert_equal(diff.sum(), 276595)
    npt.assert_equal((diff != 0).sum(), 2146)

def test_stratified_hist(sdf, pdf):
    hdf = sdf.toHandy()
    bins, _ = strat_histogram(hdf, 'Fare', bins=10, categorical=False)
    sfig, _ = hdf.stratify(['Pclass', 'Embarked']).cols['Fare'].hist()
    s64 = plot_to_pixels(sfig, (480, 640, 3))

    paxes = pdf.groupby(['Pclass', 'Embarked'])['Fare'].hist()
    pfig, axes = plt.subplots(3, 3)
    axes = [ax for row in axes for ax in row]
    idx = 0
    clauses = []
    for embarked in ['C', 'Q', 'S']:
        for pclass in [1, 2, 3]:
            clause = 'Pclass == {} and Embarked == "{}"'.format(pclass, embarked)
            clauses.append(clause)
            pdf.query(clause)['Fare'].hist(ax=axes[idx], bins=bins)
            axes[idx].grid(False)
            idx += 1

    pfig, _ = consolidate_plots(pfig, axes, 'Fare', clauses)
    p64 = plot_to_pixels(pfig, (480, 640, 3))
    npt.assert_equal(s64, p64)
