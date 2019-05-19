import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from inspect import signature
from handyspark.util import get_buckets, none2zero, ensure_list
from operator import add, itemgetter
from pyspark.ml.feature import Bucketizer
from pyspark.ml.pipeline import Pipeline
from pyspark.sql import functions as F
from matplotlib.artist import setp
import matplotlib as mpl
mpl.rc("lines", markeredgewidth=0.5)

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
            subtitle = title_fom_clause(clauses[i])
            ax.set_title(subtitle, fontdict={'fontsize': 10})
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            #if ax.colNum > 0:
            #    ax.get_yaxis().set_visible(False)
            #if ax.rowNum < (ax.numRows - 1):
            #    ax.get_xaxis().set_visible(False)
        if isinstance(title, list):
            title = ', '.join(title)
        fig.suptitle(title)
        fig.tight_layout()
        fig.subplots_adjust(top=0.85)
    return fig, axs

### Correlations
def plot_correlations(pdf, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    return sns.heatmap(round(pdf,2), annot=True, cmap="coolwarm", fmt='.2f', linewidths=.05, ax=ax)

### Scatterplot
def strat_scatterplot(sdf, col1, col2, n=30):
    stages = []
    for col in [col1, col2]:
        splits = np.linspace(*sdf.agg(F.min(col), F.max(col)).rdd.map(tuple).collect()[0], n + 1)
        bucket_name = '__{}_bucket'.format(col)
        stages.append(Bucketizer(splits=splits,
                                 inputCol=col,
                                 outputCol=bucket_name,
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

    axes = ensure_list(ax)
    clauses = sdf._handy._strata_raw_clauses
    if not len(clauses):
        clauses = [None]

    bucket_name1, bucket_name2 = '__{}_bucket'.format(col1), '__{}_bucket'.format(col2)
    strata = sdf._handy.strata_colnames
    colnames = strata + [bucket_name1, bucket_name2]
    result = model.transform(sdf).select(colnames).groupby(colnames).agg(F.count('*').alias('count')).toPandas().sort_values(by=colnames)

    splits = [bucket.getSplits() for bucket in model.stages]
    splits = [list(map(np.mean, zip(split[1:], split[:-1]))) for split in splits]
    splits1 = pd.DataFrame({bucket_name1: np.arange(0, n), col1: splits[0]})
    splits2 = pd.DataFrame({bucket_name2: np.arange(0, n), col2: splits[1]})

    df_counts = result.merge(splits1).merge(splits2)[strata + [col1, col2, 'count']].rename(columns={'count': 'Proportion'})

    df_counts.loc[:, 'Proportion'] = df_counts.Proportion.apply(lambda p: round(p / total, 4))

    for ax, clause in zip(axes, clauses):
        data = df_counts
        if clause is not None:
            data = data.query(clause)
        sns.scatterplot(data=data,
                        x=col1,
                        y=col2,
                        size='Proportion',
                        ax=ax,
                        legend=False)

    if len(axes) == 1:
        axes = axes[0]

    return axes

### Histogram
def strat_histogram(sdf, colname, bins=10, categorical=False):
    if categorical:
        result = sdf.cols[colname]._value_counts(dropna=False, raw=True)

        if hasattr(result.index, 'levels'):
            indexes = pd.MultiIndex.from_product(result.index.levels[:-1] +
                                                 [result.reset_index()[colname].unique().tolist()],
                                                 names=result.index.names)
            result = (pd.DataFrame(index=indexes)
                      .join(result.to_frame(), how='left')
                      .fillna(0)[result.name]
                      .astype(result.dtype))

        start_values = result.index.tolist()
    else:
        bucket_name = '__{}_bucket'.format(colname)
        strata = sdf._handy.strata_colnames
        colnames = strata + ensure_list(bucket_name)

        start_values = np.linspace(*sdf.agg(F.min(colname), F.max(colname)).rdd.map(tuple).collect()[0], bins + 1)
        bucketizer = Bucketizer(splits=start_values, inputCol=colname, outputCol=bucket_name, handleInvalid="skip")
        result = (bucketizer
                  .transform(sdf)
                  .select(colnames)
                  .groupby(colnames)
                  .agg(F.count('*').alias('count'))
                  .toPandas()
                  .sort_values(by=colnames))

        indexes = pd.DataFrame({bucket_name: np.arange(0, bins), 'bucket': start_values[:-1]})
        if len(strata):
            indexes = (indexes
                       .assign(key=1)
                       .merge(result[strata].drop_duplicates().assign(key=1), on='key')
                       .drop(columns=['key']))
        result = indexes.merge(result, how='left', on=strata + [bucket_name]).fillna(0)[strata + [bucket_name, 'count']]

    return start_values, result

def histogram(sdf, colname, bins=10, categorical=False, ax=None):
    strat_ax, data = sdf._get_strata()
    if data is None:
        data = strat_histogram(sdf, colname, bins, categorical)
    else:
        ax = strat_ax
    start_values, counts = data

    if ax is None:
        fig, ax = plt.subplots(1, 1)

    axes = ensure_list(ax)
    clauses = sdf._handy._strata_raw_clauses
    if not len(clauses):
        clauses = [None]

    for ax, clause in zip(axes, clauses):
        if categorical:
            pdf = counts.sort_index().to_frame()
            if clause is not None:
                pdf = pdf.query(clause).reset_index(sdf._handy.strata_colnames).drop(columns=sdf._handy.strata_colnames)
            pdf.iloc[:bins].plot(kind='bar', color='C0', legend=False, rot=0, ax=ax, title=colname)
        else:
            mid_point_bins = start_values[:-1]
            weights = counts
            if clause is not None:
                weights = counts.query(clause)
            ax.hist(mid_point_bins, bins=start_values, weights=weights['count'].values)
            ax.set_title(colname)

    if len(axes) == 1:
        axes = axes[0]

    return axes

### Boxplot
def _gen_dict(rc_name, properties):
    """ Loads properties in the dictionary from rc file if not already
    in the dictionary"""
    rc_str = 'boxplot.{0}.{1}'
    dictionary = dict()
    for prop_dict in properties:
        dictionary.setdefault(prop_dict,
                        plt.rcParams[rc_str.format(rc_name, prop_dict)])
    return dictionary

def draw_boxplot(ax, stats):
    flier_props = ['color', 'marker', 'markerfacecolor', 'markeredgecolor',
                   'markersize', 'linestyle', 'linewidth']
    default_props = ['color', 'linewidth', 'linestyle']
    boxprops = _gen_dict('boxprops', default_props)
    whiskerprops = _gen_dict('whiskerprops', default_props)
    capprops = _gen_dict('capprops', default_props)
    medianprops = _gen_dict('medianprops', default_props)
    meanprops = _gen_dict('meanprops', default_props)
    flierprops = _gen_dict('flierprops', flier_props)

    props = dict(boxprops=boxprops,
                 flierprops=flierprops,
                 medianprops=medianprops,
                 meanprops=meanprops,
                 capprops=capprops,
                 whiskerprops=whiskerprops)

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
              '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#1f77b4']
    bp = ax.bxp(stats, **props)
    ax.grid(True)
    setp(bp['boxes'], color=colors[0], alpha=1)
    setp(bp['whiskers'], color=colors[0], alpha=1)
    setp(bp['medians'], color=colors[2], alpha=1)
    return ax

def boxplot(sdf, colnames, ax=None, showfliers=True, k=1.5, precision=.0001):
    strat_ax, data = sdf._get_strata()
    if data is None:
        if ax is None:
            fig, ax = plt.subplots(1, 1)

    title_clauses = sdf._handy._strata_clauses
    if not len(title_clauses):
        title_clauses = [None]

    pdf = sdf._handy._calc_fences(colnames, k, precision)
    stats = []
    for colname in colnames:
        items, _, _ = sdf._handy._calc_bxp_stats(pdf, colname, showfliers=showfliers)
        for title_clause, item in zip(title_clauses, items):
            name = colname if len(colnames) > 1 else (title_fom_clause(title_clause) if title_clause is not None else colname)
            item.update({'label': name})

        # each list of items corresponds to a different column
        stats.append(items)

    # Stats is a list of columns, containing each a list of clauses
    if ax is not None:
        if title_clauses[0] is None:
            if len(colnames) == 1:
                stats = stats[0]
            else:
                stats = np.squeeze(stats).tolist()
        return draw_boxplot(ax, stats)
    else:
        if len(strat_ax) > 1:
            stats = [[stats[j][i] for j in range(len(stats))] for i in range(len(title_clauses))]
        return stats

def post_boxplot(axs, stats):
    new_res = []
    for ax, stat in zip(axs, stats):
        ax = draw_boxplot(ax, stat)
        new_res.append(ax)
    return new_res

def roc_curve(fpr, tpr, roc_auc, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1)

    ax.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.4f)' % roc_auc)
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic Curve')
    ax.legend(loc="lower right")
    return ax

def pr_curve(precision, recall, pr_auc, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1)

    # In matplotlib < 1.5, plt.fill_between does not have a 'step' argument
    step_kwargs = ({'step': 'post'}
                   if 'step' in signature(plt.fill_between).parameters
                   else {})
    ax.step(recall, precision, color='b', alpha=0.2, where='post', label='PR curve (area = %0.4f)' % pr_auc)
    ax.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_ylim([0.0, 1.05])
    ax.set_xlim([0.0, 1.0])
    ax.legend(loc="lower left")
    ax.set_title('Precision-Recall Curve')
    return ax