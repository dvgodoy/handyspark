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

def title_fom_clause(clause):
    return clause.replace(' and ', '\n').replace(' == ', '=').replace('"', '')

def consolidate_plots(fig, axs, title, clauses):
    axs[0].set_title(title)
    fig.tight_layout()
    if len(axs) > 1:
        assert len(axs) == len(clauses), 'Mismatched number of plots and clauses!'
        xlim = list(map(lambda ax: ax.get_xlim(), axs))
        xlim = [np.min(list(map(itemgetter(0), xlim))), np.max(list(map(itemgetter(1), xlim)))]
        ylim = list(map(lambda ax: ax.get_ylim(), axs))
        ylim = [np.min(list(map(itemgetter(0), ylim))), np.max(list(map(itemgetter(1), ylim)))]
        for i, ax in enumerate(axs):
            title = title_fom_clause(clauses[i])
            ax.set_title(title, fontdict={'fontsize': 10})
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            if ax.colNum > 0:
                ax.get_yaxis().set_visible(False)
            if ax.rowNum < (ax.numRows - 1):
                ax.get_xaxis().set_visible(False)
        if isinstance(title, list):
            title = ', '.join(title)
        fig.suptitle(title)
        fig.tight_layout()
        fig.subplots_adjust(top=0.9)
    return fig

### Correlations
def correlations(sdf, colnames, ax=None, plot=True):
    correlations = Statistics.corr(sdf.select(colnames).dropna().rdd.map(lambda row: row[0:]))
    pdf = pd.DataFrame(correlations, columns=colnames, index=colnames)
    if plot:
        sns.heatmap(round(pdf,2), annot=True, cmap="coolwarm", fmt='.2f', linewidths=.05, ax=ax)
    return pdf

### Scatterplot
def strat_scatterplot(sdf, col1, col2, n=30):
    stages = []
    for col in [col1, col2]:
        splits = get_buckets(sdf.select(col).rdd.map(itemgetter(0)), n)
        stages.append(Bucketizer(splits=splits,
                                 inputCol=col,
                                 outputCol="__{}_bucket".format(col),
                                 handleInvalid="skip"))

    pipeline = Pipeline(stages=stages)
    model = pipeline.fit(sdf)
    return model, sdf.count()

def scatterplot(sdf, col1, col2, n=30, ax=None):
    strat_ax, data = sdf._get_strata()
    if data is None:
        data = strat_scatterplot(sdf, col1, col2, n)
    else:
        ax = strat_ax
    model, total = data

    if ax is None:
        fig, ax = plt.subplots(1, 1)

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

    df_counts.loc[:, 'Proportion'] = df_counts.Proportion.apply(lambda p: round(p / total, 4))

    sns.scatterplot(data=df_counts,
                    x=col1,
                    y=col2,
                    size='Proportion',
                    ax=ax,
                    legend=False)
    return ax

### Histogram
def strat_histogram(sdf, colname, bins=10, categorical=False):
    if categorical:
        start_values = (sdf.select(colname)
                        .rdd
                        .map(lambda row: (itemgetter(0)(row), 1))
                        .reduceByKey(add)
                        .sortBy(itemgetter(1), ascending=False)
                        .collect())
        counts = list(map(itemgetter(1), start_values))
        start_values = list(map(itemgetter(0), start_values))
    else:
        start_values, counts = sdf.select(colname).rdd.map(itemgetter(0)).histogram(bins)
    return start_values, counts

def histogram(sdf, colname, bins=10, categorical=False, ax=None):
    strat_ax, data = sdf._get_strata()
    if data is None:
        data = strat_histogram(sdf, colname, bins, categorical)
    else:
        ax = strat_ax
    start_values, counts = data

    if ax is None:
        fig, ax = plt.subplots(1, 1)

    if categorical:
        values = dict(sdf.select(colname)
                      .rdd
                      .map(lambda row: (itemgetter(0)(row), 1))
                      .reduceByKey(add)
                      .sortBy(itemgetter(1), ascending=False)
                      .collect())

        values = list(map(lambda k: (k, values.get(k, 0)), start_values))

        pdf = pd.Series(map(itemgetter(1), values),
                        index=map(itemgetter(0), values),
                        name=colname).sort_index().to_frame().iloc[:bins]
        pdf.plot(kind='bar', color='C0', legend=False, rot=0, ax=ax, title=colname)
        return ax
    else:
        _, counts = sdf.select(colname).rdd.map(itemgetter(0)).histogram(start_values)
        mid_point_bins = start_values[:-1]
        ax.hist(mid_point_bins, bins=start_values, weights=counts)
        ax.set_title(colname)
        return ax

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

### Boxplot
def _calc_tukey(col_summ):
    q1, q3 = float(col_summ['25%']), float(col_summ['75%'])
    iqr = q3 - q1
    lfence = q1 - (1.5 * iqr)
    ufence = q3 + (1.5 * iqr)
    return lfence, ufence

def boxplot(sdf, colnames, ax=None):
    strat_ax, data = sdf._get_strata()
    if data is None:
        if ax is None:
            fig, ax = plt.subplots(1, 1)

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

    if ax is not None:
        ax.bxp(stats)
        return ax
    else:
        return stats

def post_boxplot(axs, stats, clauses):
    if len(axs) == len(stats):
        new_res = []
        for ax, stats in zip(axs, stats):
            ax.bxp(stats)
            new_res.append(ax)
    else:
        ax = axs[0]
        items = []
        for clause, stats in zip(clauses, stats):
            label = title_fom_clause(clause)
            stats[0].update({'label': label})
            items.append(stats[0])
        ax.bxp(items)
        new_res = [ax]
    return new_res
