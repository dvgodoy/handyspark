import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from handyspark.util import get_buckets
from operator import add, itemgetter
from pyspark.ml.feature import Bucketizer
from pyspark.ml.pipeline import Pipeline
from pyspark.mllib.stat import Statistics
from pyspark.sql import functions as F

### Correlations
def correlations(sdf, colnames, ax=None, plot=True):
    correlations = Statistics.corr(sdf.select(colnames).rdd.map(lambda row: row[0:]))
    pdf = pd.DataFrame(correlations, columns=colnames, index=colnames)
    if plot:
        sns.heatmap(round(pdf,2), annot=True, cmap="coolwarm", fmt='.2f', linewidths=.05, ax=ax)
    return pdf

### Scatterplot
def scatterplot(sdf, col1, col2, n=30, ax=None):
    stages = []
    for col in [col1, col2]:
        splits = get_buckets(sdf.select(col).rdd.map(itemgetter(0)), n)
        stages.append(Bucketizer(splits=splits,
                                 inputCol=col,
                                 outputCol="__{}_bucket".format(col),
                                 handleInvalid="skip"))

    pipeline = Pipeline(stages=stages)
    model = pipeline.fit(sdf)
    counts = (model
              .transform(sdf.select(col1, col2).dropna())
              .select(*("__{}_bucket".format(col) for col in (col1, col2)))
              .rdd
              .map(lambda row: (row[0:], 1))
              .reduceByKey(add)
              .collect())
    splits = [bucket.getSplits() for bucket in model.stages]
    splits = [list(map(np.mean, zip(split[1:], split[:-1]))) for split in splits]

    df_counts = pd.DataFrame([(splits[0][int(v[0][0])],
                               splits[1][int(v[0][1])],
                               v[1]) for v in counts],
                             columns=[col1, col2, 'Proportion'])

    #df_counts = (model
    #             .transform(sdf.select(col1, col2).dropna())
    #             .select(*("__{}_bucket".format(col) for col in (col1, col2)))
    #             .withColumnRenamed("__{}_bucket".format(col1), col1)
    #             .withColumnRenamed("__{}_bucket".format(col2), col2)
    #             .crosstab(col1, col2)
    #             .withColumnRenamed('{}_{}'.format(col1, col2), col1)
    #             .toPandas()
    #             .melt(id_vars=col1, var_name=col2, value_name='Count')
    #             .query('Count > 0'))

    #df_counts.loc[:, col1] = pd.to_numeric(df_counts[col1])
    #df_counts.loc[:, col2] = pd.to_numeric(df_counts[col2])
    df_counts.loc[:, 'Proportion'] /= df_counts.Proportion.sum()

    sns.scatterplot(data=df_counts,
                    x=col1,
                    y=col2,
                    size='Proportion',
                    ax=ax)
    return

### Histogram
def histogram(sdf, colname, bins=10, categorical=False, ax=None):
    if categorical:
        values = (sdf.select(colname)
                  .rdd
                  .map(lambda row: (itemgetter(0)(row), 1))
                  .reduceByKey(add)
                  .sortBy(itemgetter(1), ascending=False)
                  .collect())

        pdf = pd.Series(map(itemgetter(1), values),
                        index=map(itemgetter(0), values),
                        name=colname).sort_index().to_frame().iloc[:bins]
        pdf.plot(kind='bar', color='C0', legend=False, rot=0, ax=ax, title=colname)
        return pdf
    else:
        start_values, counts = sdf.select(colname).rdd.map(itemgetter(0)).histogram(bins)
        mid_point_bins = start_values[:-1]
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        ax.hist(mid_point_bins, bins=start_values, weights=counts)
        ax.set_title(colname)
        return start_values, counts

### Stratified Histogram
def stratified_histogram(sdf, colname, strat_colname, strat_values, ax=None):
    buckets = get_buckets(sdf.select(colname).rdd.map(itemgetter(0)), 20)
    for value in strat_values:
        start_values, counts = (sdf
                                .select(colname)
                                .filter('{} == {}'.format(strat_colname, value))
                                .rdd
                                .map(itemgetter(0))
                                .histogram(buckets))
        sns.distplot(start_values[:len(counts)],
                     bins=start_values,
                     color='C{}'.format(value - 1),
                     norm_hist=True,
                     kde=False,
                     hist_kws={"weights":counts},
                     label='{}'.format(value),
                     ax=ax)
    ax.set_legend()
    return

def _calc_tukey(col_summ):
    q1, q3 = float(col_summ['25%']), float(col_summ['75%'])
    iqr = q3 - q1
    lfence = q1 - (1.5 * iqr)
    ufence = q3 + (1.5 * iqr)
    return lfence, ufence

### Boxplot
def boxplot(sdf, colnames, ax=None):
    pdf = sdf.select(colnames).notHandy.summary().toPandas().set_index('summary')
    pdf.loc['fence', :] = pdf.apply(_calc_tukey)

    # faster than stats()
    def minmax(a, b):
        return min(a[0], b[0]), max(a[1], b[1])

    stats = []
    for colname in colnames:
        col_summ = pdf[colname]
        lfence, ufence = col_summ.fence

        outlier = sdf.withColumn('__{}_outlier'.format(colname),
                                 ~F.col(colname).between(lfence, ufence))

        minv, maxv = (outlier
                      .filter('not __{}_outlier'.format(colname))
                      .select(colname)
                      .rdd
                      .map(lambda x: (x[0], x[0]))
                      .reduce(minmax))

        #minmax = (outlier.filter('not __{}_outlier'.format(colname))
        #          .select(F.min(colname).alias('__{}_min'.format(colname)),
        #                  F.max(colname).alias('__{}_max'.format(colname)))
        #          .collect()[0])
        #minv = minmax['__{}_min'.format(colname)]
        #maxv = minmax['__{}_max'.format(colname)]

        fliers = (outlier
                  .filter('__{}_outlier'.format(colname))
                  .select(colname)
                  .rdd
                  .map(itemgetter(0))
                  .sortBy(lambda v: -abs(v))
                  .take(100))

        item = {'label': colname,
                'mean': float(col_summ['mean']),
                'med': float(col_summ['50%']),
                'q1': float(col_summ['25%']),
                'q3': float(col_summ['75%']),
                'whislo': minv,
                'whishi': maxv,
                'fliers': fliers}

        stats.append(item)

    ax.bxp(stats)
    return
