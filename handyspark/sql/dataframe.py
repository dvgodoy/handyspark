from copy import deepcopy
from handyspark.ml.base import HandyTransformers
from handyspark.plot import histogram, boxplot, scatterplot, strat_scatterplot, strat_histogram,\
    consolidate_plots, post_boxplot
from handyspark.sql.pandas import HandyPandas
from handyspark.sql.transform import _MAPPING, HandyTransform
from handyspark.util import HandyException, dense_to_array, disassemble, ensure_list, check_columns, \
    none2default
import inspect
from matplotlib.axes import Axes
from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np
from operator import itemgetter, add
import pandas as pd
from pyspark.ml.stat import Correlation
from pyspark.ml.feature import Bucketizer
from pyspark.mllib.stat import Statistics
from pyspark.sql import DataFrame, GroupedData, Window, functions as F, Column, Row
from pyspark.ml.feature import VectorAssembler, StandardScaler, PCA
from pyspark.ml.pipeline import Pipeline
from scipy.stats import chi2
from scipy.linalg import inv

def toHandy(self):
    """Converts Spark DataFrame into HandyFrame.
    """
    return HandyFrame(self)

def notHandy(self):
    return self

DataFrame.toHandy = toHandy
DataFrame.notHandy = notHandy

def agg(f):
    f.__is_agg = True
    return f

def inccol(f):
    f.__is_inccol = True
    return f

class Handy(object):
    def __init__(self, df):
        self._df = df

        # classification
        self._is_classification = False
        self._nclasses = None
        self._classes = None

        # transformers
        self._imputed_values = {}
        self._fenced_values = {}

        # groups / strata
        self._group_cols = None
        self._strata = None
        self._strata_object = None
        self._strata_plot = None

        self._clear_stratification()
        self._safety_limit = 1000
        self._safety = True

        self._update_types()

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k not in ['_df', '_strata_object', '_strata_plot']:
                setattr(result, k, deepcopy(v, memo))
        return result

    def __getitem__(self, *args):
        if isinstance(args[0], tuple):
            args = args[0]
        item = args[0]
        n = 20
        if len(args) > 1:
            n = args[1]
            if n is None:
                n = -1

        if isinstance(item, int):
            idx = item + (len(self._group_cols) if self._group_cols is not None else 0)
            assert idx < len(self._df.columns), "Invalid column index {}".format(idx)
            item = list(self._df.columns)[idx]

        if isinstance(item, str):
            if self._group_cols is None or len(self._group_cols) == 0:
                res = self._take_array(item, n)
                if res.ndim > 1:
                    res = res.tolist()
                res = pd.Series(res, name=item)
                if self._strata is not None:
                    strata = list(map(lambda v: v[1].to_dict(), self.strata.iterrows()))
                    if len(strata) == len(res):
                        res = pd.concat([pd.DataFrame(strata), res], axis=1).set_index(self._strata).sort_index()
                return res
            else:
                check_columns(self._df, list(self._group_cols) + [item])
                pdf = self._df.notHandy().select(list(self._group_cols) + [item])
                if n != -1:
                    pdf = pdf.limit(n)
                res = pdf.toPandas().set_index(list(self._group_cols)).sort_index()[item]
                return res

    @property
    def stages(self):
        return (len(list(filter(lambda v: '+' == v,
                                map(lambda s: s.strip()[0],
                                    self._df.rdd.toDebugString().decode().split('\n'))))) + 1)

    @property
    def statistics_(self):
        return self._imputed_values

    @property
    def fences_(self):
        return self._fenced_values

    @property
    def is_classification(self):
        return self._is_classification

    @property
    def classes(self):
        return self._classes

    @property
    def nclasses(self):
        return self._nclasses

    @property
    def response(self):
        return self._response

    @property
    def ncols(self):
        return len(self._types)

    @property
    def nrows(self):
        return self._df.count()

    @property
    def shape(self):
        return (self.nrows, self.ncols)

    @property
    def strata(self):
        if self._strata is not None:
            return pd.DataFrame(data=self._strata_combinations, columns=self._strata)

    @property
    def strata_colnames(self):
        if self._strata is not None:
            return list(map(str, ensure_list(self._strata)))
        else:
            return []

    def _stratify(self, strata):
        return HandyStrata(self, strata)

    def _clear_stratification(self):
        self._strata = None
        self._strata_object = None
        self._strata_plot = None
        self._strata_combinations = []
        self._strata_raw_combinations = []
        self._strata_clauses = []
        self._strata_raw_clauses = []
        self._n_cols = 1
        self._n_rows = 1

    def _set_stratification(self, strata, raw_combinations, raw_clauses, combinations, clauses):
        if strata is not None:
            assert len(combinations[0]) == len(strata), "Mismatched number of combinations and strata!"
            self._strata = strata
            self._strata_raw_combinations = raw_combinations
            self._strata_raw_clauses = raw_clauses
            self._strata_combinations = combinations
            self._strata_clauses = clauses
            self._n_cols = len(set(map(itemgetter(0), combinations)))
            try:
                self._n_rows = len(set(map(itemgetter(1), combinations)))
            except IndexError:
                self._n_rows = 1

    def _build_strat_plot(self, n_rows, n_cols, **kwargs):
        fig, axs = plt.subplots(n_rows, n_cols, **kwargs)
        if n_rows == 1:
            axs = [axs]
            if n_cols == 1:
                axs = [axs]
        self._strata_plot = (fig, [ax for col in np.transpose(axs) for ax in col])

    def _update_types(self):
        self._types = list(map(lambda t: (t.name, t.dataType.typeName()), self._df.schema.fields))

        self._numerical = list(map(itemgetter(0), filter(lambda t: t[1] in ['byte', 'short', 'integer', 'long',
                                                                            'float', 'double'], self._types)))
        self._continuous = list(map(itemgetter(0), filter(lambda t: t[1] in ['double', 'float'], self._types)))
        self._categorical = list(map(itemgetter(0), filter(lambda t: t[1] in ['byte', 'short', 'integer', 'long',
                                                                              'boolan', 'string'], self._types)))
        self._array = list(map(itemgetter(0), filter(lambda t: t[1] in ['array', 'map'], self._types)))
        self._string = list(map(itemgetter(0), filter(lambda t: t[1] in ['string'], self._types)))

    def _take_array(self, colname, n):
        check_columns(self._df, colname)
        datatype = self._df.notHandy().select(colname).schema.fields[0].dataType.typeName()
        rdd = self._df.notHandy().select(colname).rdd.map(itemgetter(0))

        if n == -1:
            data = rdd.collect()
        else:
            data = rdd.take(n)

        return np.array(data, dtype=_MAPPING.get(datatype, 'object'))

    def _value_counts(self, colnames, dropna=True, raw=False):
        colnames = ensure_list(colnames)
        strata = self.strata_colnames
        colnames = strata + colnames

        check_columns(self._df, colnames)
        data = self._df.notHandy().select(colnames)
        if dropna:
            data = data.dropna()

        values = (data.groupby(colnames).agg(F.count('*').alias('value_counts'))
                  .toPandas().set_index(colnames).sort_index()['value_counts'])

        if not raw:
            for level, col in enumerate(ensure_list(self._strata)):
                if not isinstance(col, str):
                    values.index.set_levels(pd.Index(col._clauses[1:-1]), level=level, inplace=True)
                    values.index.set_names(col.colname, level=level, inplace=True)

        return values

    def _fillna(self, target, values):
        assert isinstance(target, DataFrame), "Target must be a DataFrame"

        items = values.items()
        for colname, v in items:
            if isinstance(v, dict):
                clauses = v.keys()
                whens = ' '.join(['WHEN (({clause}) AND (isnan({col}) OR isnull({col}))) THEN {quote}{filling}{quote}'
                                 .format(clause=clause, col=colname, filling=v[clause],
                                         quote='"' if isinstance(v[clause], str) else '')
                                   for clause in clauses])
            else:
                whens = ('WHEN (isnan({col}) OR isnull({col})) THEN {quote}{filling}{quote}'
                         .format(col=colname, filling=v,
                                 quote='"' if isinstance(v, str) else ''))

            expression = F.expr('CASE {expr} ELSE {col} END'.format(expr=whens, col=colname))
            target = target.withColumn(colname, expression)

        return target

    def __stat_to_dict(self, colname, stat):
        if len(self._strata_clauses):
            if isinstance(stat, pd.Series):
                stat = stat.to_frame(colname)
            return {clause: stat.query(raw_clause)[colname].iloc[0]
                    for clause, raw_clause in zip(self._strata_clauses, self._strata_raw_clauses)}
        else:
            return stat[colname]

    def _fill_values(self, continuous, categorical, strategy):
        values = {}
        colnames = list(map(itemgetter(0), filter(lambda t: t[1] == 'mean', zip(continuous, strategy))))
        values.update(dict([(col, self.__stat_to_dict(col, self.mean(col))) for col in colnames]))

        colnames = list(map(itemgetter(0), filter(lambda t: t[1] == 'median', zip(continuous, strategy))))
        values.update(dict([(col, self.__stat_to_dict(col, self.median(col))) for col in colnames]))

        values.update(dict([(col, self.__stat_to_dict(col, self.mode(col)))
                            for col in categorical if col in self._categorical]))
        return values

    def __fill_self(self, continuous, categorical, strategy):
        continuous = ensure_list(continuous)
        categorical = ensure_list(categorical)
        check_columns(self._df, continuous + categorical)

        strategy = none2default(strategy, 'mean')

        if continuous == ['all']:
            continuous = self._continuous
        if categorical == ['all']:
            categorical = self._categorical

        if isinstance(strategy, (list, tuple)):
            assert len(continuous) == len(strategy), "There must be a strategy to each column."
        else:
            strategy = [strategy] * len(continuous)

        values = self._fill_values(continuous, categorical, strategy)
        self._imputed_values.update(values)
        res = HandyFrame(self._fillna(self._df, values), self)
        return res

    def _dense_to_array(self, colname, array_colname):
        check_columns(self._df, colname)
        res = dense_to_array(self._df.notHandy(), colname, array_colname)
        return HandyFrame(res, self)

    def _agg(self, name, func, colnames):
        colnames = none2default(colnames, self._df.columns)
        colnames = ensure_list(colnames)
        check_columns(self._df, self.strata_colnames + [col for col in colnames if not isinstance(col, Column)])
        if func is None:
            func = getattr(F, name)

        res = (self._df.notHandy()
               .groupby(self.strata_colnames)
               .agg(*(func(col).alias(str(col)) for col in colnames if str(col) not in self.strata_colnames))
               .toPandas())

        if len(res) == 1:
            res = res.iloc[0]
            res.name = name
        return res

    def _calc_fences(self, colnames, k=1.5, precision=.01):
        colnames = none2default(colnames, self._numerical)
        colnames = ensure_list(colnames)
        check_columns(self._df, colnames)
        colnames = [col for col in colnames if col in self._numerical]
        strata = self.strata_colnames

        pdf = (self._df.notHandy()
               .groupby(strata)
               .agg(F.count(F.lit(1)).alias('nrows'),
                    *[F.expr('approx_percentile({}, {}, {})'.format(c, q, 1./precision)).alias('{}_{}%'.format(c, int(q * 100)))
                      for q in [.25, .50, .75] for c in colnames],
                    *[F.mean(c).alias('{}_mean'.format(c)) for c in colnames]).toPandas())

        for col in colnames:
            pdf.loc[:, '{}_iqr'.format(col)] = pdf.loc[:, '{}_75%'.format(col)] - pdf.loc[:, '{}_25%'.format(col)]
            pdf.loc[:, '{}_lfence'.format(col)] = pdf.loc[:, '{}_25%'.format(col)] - k * pdf.loc[:, '{}_iqr'.format(col)]
            pdf.loc[:, '{}_ufence'.format(col)] = pdf.loc[:, '{}_75%'.format(col)] + k * pdf.loc[:, '{}_iqr'.format(col)]

        return pdf

    def _calc_mahalanobis_distance(self, colnames, output_col='__mahalanobis'):
        """Computes Mahalanobis distance from origin
        """
        sdf = self._df.notHandy()
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
                    .drop('__features', '__scaled', '__array_scaled'))
        return distance

    def _set_mahalanobis_outliers(self, colnames, critical_value=.999,
                                  input_col='__mahalanobis', output_col='__outlier'):
        """Compares Mahalanobis distances to critical values using
         Chi-Squared distribution to identify possible outliers.
        """
        distance = self._calc_mahalanobis_distance(colnames)
        # Computes critical value
        critical_value = chi2.ppf(critical_value, len(colnames))
        # Computes Mahalanobis distance and flags as outliers all elements above critical value
        outlier = (distance.withColumn(output_col, F.col(input_col) > critical_value))
        return outlier

    def _calc_bxp_stats(self, fences_df, colname, showfliers=False):
        strata = self.strata_colnames
        clauses = self._strata_raw_clauses
        if not len(clauses):
            clauses = [None]

        qnames = ['25%', '50%', '75%', 'mean', 'lfence', 'ufence']
        col_summ = fences_df[strata + ['{}_{}'.format(colname, q) for q in qnames] + ['nrows']]
        col_summ.columns = strata + qnames + ['nrows']
        if len(strata):
            col_summ = col_summ.set_index(strata)
        lfence, ufence = col_summ[['lfence']], col_summ[['ufence']]

        expression = None
        for clause in clauses:
            if clause is not None:
                partial = F.col(colname).between(lfence.query(clause).iloc[0, 0], ufence.query(clause).iloc[0, 0])
                partial &= F.expr(clause)
            else:
                partial = F.col(colname).between(lfence.iloc[0, 0], ufence.iloc[0, 0])

            if expression is None:
                expression = partial
            else:
                expression |= partial

        outlier = self._df.notHandy().withColumn('__{}_outlier'.format(colname), ~expression)
        minmax = (outlier
                  .filter('not __{}_outlier'.format(colname))
                  .groupby(strata)
                  .agg(F.min(colname).alias('min'),
                       F.max(colname).alias('max'))
                  .toPandas())

        if len(strata):
            minmax = [minmax.query(clause).iloc[0][['min', 'max']].values for clause in clauses]
        else:
            minmax = [minmax.iloc[0][['min', 'max']].values]

        fliers_df = outlier.filter('__{}_outlier'.format(colname))
        fliers_df = [fliers_df.filter(clause) for clause in clauses] if len(strata) else [fliers_df]
        fliers_count = [df.count() for df in fliers_df]

        if showfliers:
            fliers = [(df
                       .select(F.abs(F.col(colname)).alias(colname))
                       .orderBy(F.desc(colname))
                       .limit(1000)
                       .toPandas()[colname].values) for df in fliers_df]
        else:
            fliers = [[]] * len(clauses)

        stats = []  # each item corresponds to a different clause - all items belong to the same column
        nrows = []
        for clause, whiskers, outliers in zip(clauses, minmax, fliers):
            summary = col_summ
            if clause is not None:
                summary = summary.query(clause)
            item = {'mean': summary['mean'].values[0],
                    'med': summary['50%'].values[0],
                    'q1': summary['25%'].values[0],
                    'q3': summary['75%'].values[0],
                    'whislo': whiskers[0],
                    'whishi': whiskers[1],
                    'fliers': outliers}
            stats.append(item)
            nrows.append(summary['nrows'].values[0])

        if not len(nrows):
            nrows = summary['nrows'].values[0]

        return stats, fliers_count, nrows

    def set_response(self, colname):
        check_columns(self._df, colname)
        self._response = colname
        if colname is not None:
            if colname not in self._continuous:
                self._is_classification = True
                self._classes = self._df.notHandy().select(colname).rdd.map(itemgetter(0)).distinct().collect()
                self._nclasses = len(self._classes)

        return self

    def disassemble(self, colname, new_colnames=None):
        check_columns(self._df, colname)
        res = disassemble(self._df.notHandy(), colname, new_colnames)
        return HandyFrame(res, self)

    def to_metrics_RDD(self, prob_col, label):
        check_columns(self._df, [prob_col, label])
        return self.disassemble(prob_col).select('{}_1'.format(prob_col), F.col(label).cast('double')).rdd.map(tuple)

    def corr(self, colnames=None, method='pearson'):
        colnames = none2default(colnames, self._numerical)
        colnames = ensure_list(colnames)
        check_columns(self._df, colnames)
        colnames = [col for col in colnames if col in self._numerical]
        if self._strata is not None:
            colnames = sorted([col for col in colnames if col not in self.strata_colnames])

        correlations = Statistics.corr(self._df.notHandy().select(colnames).dropna().rdd.map(lambda row: row[0:]), method=method)
        pdf = pd.DataFrame(correlations, columns=colnames, index=colnames)
        return pdf

    def fill(self, *args, continuous=None, categorical=None, strategy=None):
        if len(args) and isinstance(args[0], DataFrame):
            return self._fillna(args[0], self._imputed_values)
        else:
            return self.__fill_self(continuous=continuous, categorical=categorical, strategy=strategy)

    @agg
    def isnull(self, ratio=False):
        def func(colname):
            return F.sum(F.isnull(colname).cast('int')).alias(colname)

        name = 'missing'
        if ratio:
            name += '(ratio)'
        missing = self._agg(name, func, self._df.columns)

        if ratio:
            nrows = self._agg('nrows', F.sum, F.lit(1))
            if isinstance(missing, pd.Series):
                missing = missing / nrows["Column<b'1'>"]
            else:
                missing.iloc[:, 1:] = missing.iloc[:, 1:].values / nrows["Column<b'1'>"].values.reshape(-1, 1)

        if len(self.strata_colnames):
            missing = missing.set_index(self.strata_colnames).T.unstack()
            missing.name = name

        return missing

    @agg
    def nunique(self, colnames=None):
        res = self._agg('nunique', F.approx_count_distinct, colnames)
        if len(self.strata_colnames):
            res = res.set_index(self.strata_colnames).T.unstack()
            res.name = 'nunique'
        return res

    def outliers(self, colnames=None, ratio=False, method='tukey', **kwargs):
        colnames = none2default(colnames, self._numerical)
        colnames = ensure_list(colnames)
        check_columns(self._df, colnames)
        colnames = [col for col in colnames if col in self._numerical]

        res = None
        if method == 'tukey':
            outliers = []
            try:
                k = float(kwargs['k'])
            except KeyError:
                k = 1.5
            fences_df = self._calc_fences(colnames, k=k, precision=.01)

            index = fences_df[self.strata_colnames].set_index(self.strata_colnames).index \
                if len(self.strata_colnames) else None

            for colname in colnames:
                stats, counts, nrows = self._calc_bxp_stats(fences_df, colname, showfliers=False)
                outliers.append(pd.Series(counts, index=index, name=colname))
                if ratio:
                    outliers[-1] /= nrows

            res = pd.DataFrame(outliers).unstack()
            if not len(self.strata_colnames):
                res = res.droplevel(0)
            name = 'outliers'
            if ratio:
                name += '(ratio)'
            res.name = name

        return res

    def get_outliers(self, colnames=None, critical_value=.999):
        colnames = none2default(colnames, self._numerical)
        colnames = ensure_list(colnames)
        check_columns(self._df, colnames)
        colnames = [col for col in colnames if col in self._numerical]

        outliers = self._set_mahalanobis_outliers(colnames, critical_value)
        df = outliers.filter('__outlier').orderBy(F.desc('__mahalanobis')).drop('__outlier', '__mahalanobis')
        return HandyFrame(df, self)

    def remove_outliers(self, colnames=None, critical_value=.999):
        colnames = none2default(colnames, self._numerical)
        colnames = ensure_list(colnames)
        check_columns(self._df, colnames)
        colnames = [col for col in colnames if col in self._numerical]

        outliers = self._set_mahalanobis_outliers(colnames, critical_value)
        df = outliers.filter('not __outlier').drop('__outlier', '__mahalanobis')
        return HandyFrame(df, self)

    def fence(self, colnames, k=1.5):
        colnames = ensure_list(colnames)
        check_columns(self._df, colnames)
        colnames = [col for col in colnames if col in self._numerical]

        pdf = self._calc_fences(colnames, k=k)
        if len(self.strata_colnames):
            pdf = pdf.set_index(self.strata_colnames)

        df = self._df.notHandy()
        for colname in colnames:
            lfence, ufence = pdf.loc[:, ['{}_lfence'.format(colname)]], pdf.loc[:, ['{}_ufence'.format(colname)]]
            if len(self._strata_raw_clauses):
                whens1 = ' '.join(['WHEN ({clause}) THEN greatest({col}, {fence})'.format(clause=clause,
                                                                                          col=colname,
                                                                                          fence=lfence.query(clause).iloc[0, 0])
                                   for clause in self._strata_raw_clauses])
                whens2 = ' '.join(['WHEN ({clause}) THEN least({col}, {fence})'.format(clause=clause,
                                                                                       col=colname,
                                                                                       fence=ufence.query(clause).iloc[0, 0])
                                   for clause in self._strata_raw_clauses])
                expression1 = F.expr('CASE {} END'.format(whens1))
                expression2 = F.expr('CASE {} END'.format(whens2))
                self._fenced_values.update({colname: {clause: [lfence.query(clause).iloc[0, 0],
                                                               ufence.query(clause).iloc[0, 0]]
                                                      for clause in self._strata_clauses}})
            else:
                self._fenced_values.update({colname: [lfence.iloc[0, 0], ufence.iloc[0, 0]]})

                expression1 = F.expr('greatest({col}, {fence})'.format(col=colname, fence=lfence.iloc[0, 0]))
                expression2 = F.expr('least({col}, {fence})'.format(col=colname, fence=ufence.iloc[0, 0]))
            df = df.withColumn(colname, expression1).withColumn(colname, expression2)

        return HandyFrame(df.select(self._df.columns), self)

    @inccol
    def value_counts(self, colnames, dropna=True):
        return self._value_counts(colnames, dropna)

    @inccol
    def mode(self, colname):
        check_columns(self._df, [colname])

        if self._strata is None:
            values = (self._df.notHandy().select(colname).dropna()
                      .groupby(colname).agg(F.count('*').alias('mode'))
                      .orderBy(F.desc('mode')).limit(1)
                      .toPandas()[colname][0])
            return pd.Series(values, index=[colname], name='mode')
        else:
            strata = self.strata_colnames
            colnames = strata + [colname]
            values = (self._df.notHandy().select(colnames).dropna()
                      .groupby(colnames).agg(F.count('*').alias('mode'))
                      .withColumn('order', F.row_number().over(Window.partitionBy(strata).orderBy(F.desc('mode'))))
                      .filter('order == 1').drop('order')
                      .toPandas().set_index(strata).sort_index()[colname])
            values.name = 'mode'
            return values

    @inccol
    def entropy(self, colnames):
        colnames = ensure_list(colnames)
        check_columns(self._df, colnames)
        sdf = self._df.notHandy()
        n = sdf.count()
        entropy = []
        for colname in colnames:
            if colname in self._categorical:
                res = (self._df
                       .groupby(self.strata_colnames + [colname])
                       .agg(F.count('*').alias('value_counts')).withColumn('probability', F.col('value_counts') / n)
                       .groupby(self.strata_colnames)
                       .agg(F.sum(F.expr('-log2(probability) * probability')).alias(colname))
                       .safety_off()
                       .cols[self.strata_colnames + [colname]][:])

                if len(self.strata_colnames):
                    res.set_index(self.strata_colnames, inplace=True)
                    res = res.unstack()
                else:
                    res = res[colname]
                    res.index = [colname]
            else:
                res = pd.Series(None, index=[colname])
            res.name = 'entropy'
            entropy.append(res)
        return pd.concat(entropy).sort_index()

    @inccol
    def mutual_info(self, colnames):
        def distribution(sdf, colnames):
            return sdf.groupby(colnames).agg(F.count('*').alias('__count'))

        check_columns(self._df, colnames)
        n = len(colnames)
        probs = []
        sdf = self._df.notHandy()
        for i in range(n):
            probs.append(distribution(sdf, self.strata_colnames + [colnames[i]]))

        if len(self.strata_colnames):
            nrows = sdf.groupby(self.strata_colnames).agg(F.count('*').alias('__n'))
        else:
            nrows = sdf.count()

        entropies = self.entropy(colnames)
        res = []
        for i in range(n):
            for j in range(i, n):
                if i == j:
                    mi = pd.Series(entropies[colnames[i]], name='mi').to_frame()
                else:
                    tdf = distribution(sdf, self.strata_colnames + [colnames[i], colnames[j]])
                    if len(self.strata_colnames):
                        tdf = tdf.join(nrows, on=self.strata_colnames)
                    else:
                        tdf = tdf.withColumn('__n', F.lit(nrows))
                    tdf = tdf.join(probs[i].toDF(*self.strata_colnames, colnames[i], '__count0'), on=self.strata_colnames + [colnames[i]])
                    tdf = tdf.join(probs[j].toDF(*self.strata_colnames, colnames[j], '__count1'), on=self.strata_colnames + [colnames[j]])
                    mi = (tdf
                          .groupby(self.strata_colnames)
                          .agg(F.sum(F.expr('log2(__count * __n / (__count0 * __count1)) * __count / __n')).alias('mi'))
                          .toPandas())

                    if len(self.strata_colnames):
                        mi.set_index(self.strata_colnames, inplace=True)

                    res.append(mi.assign(ci=colnames[j], cj=colnames[i]))

                res.append(mi.assign(ci=colnames[i], cj=colnames[j]))

        res = pd.concat(res).set_index(['ci', 'cj'], append=len(self.strata_colnames)).sort_index()
        res = pd.pivot_table(res, index=self.strata_colnames + ['ci'], columns=['cj'])
        res.index.names = self.strata_colnames + ['']
        res.columns = res.columns.droplevel(0).rename('')
        return res

    @agg
    def mean(self, colnames):
        return self._agg('mean', F.mean, colnames)

    @agg
    def min(self, colnames):
        return self._agg('min', F.min, colnames)

    @agg
    def max(self, colnames):
        return self._agg('max', F.max, colnames)

    @agg
    def percentile(self, colnames, perc=50, precision=.01):
        def func(c):
            return F.expr('approx_percentile({}, {}, {})'.format(c, perc/100., 1./precision))
        try:
            name = {25: 'q1', 50: 'median', 75: 'q3'}[perc]
        except KeyError:
            name = 'percentile_{}'.format(perc)
        return self._agg(name, func, colnames)

    @agg
    def median(self, colnames, precision=.01):
        return self.percentile(colnames, 50, precision)

    @agg
    def stddev(self, colnames):
        return self._agg('stddev', F.stddev, colnames)

    @agg
    def var(self, colnames):
        return self._agg('var', F.stddev, colnames) ** 2

    @agg
    def q1(self, colnames, precision=.01):
        return self.percentile(colnames, 25, precision)

    @agg
    def q3(self, colnames, precision=.01):
        return self.percentile(colnames, 75, precision)

    ### Boxplot functions
    def _strat_boxplot(self, colnames, **kwargs):
        n_rows = n_cols = 1
        kwds = deepcopy(kwargs)
        for kw in ['showfliers', 'precision']:
            try:
                del kwds[kw]
            except KeyError:
                pass
        if isinstance(colnames, (tuple, list)) and (len(colnames) > 1):
            n_rows = self._n_rows
            n_cols = self._n_cols
        self._build_strat_plot(n_rows, n_cols, **kwds)
        return None

    @inccol
    def boxplot(self, colnames, ax=None, showfliers=True, k=1.5, precision=.01, **kwargs):
        colnames = ensure_list(colnames)
        check_columns(self._df, colnames)
        colnames = [col for col in colnames if col in self._numerical]
        assert len(colnames), "Only numerical columns can be plot!"
        return boxplot(self._df, colnames, ax, showfliers, k, precision)

    def _post_boxplot(self, res):
        return post_boxplot(self._strata_plot[1], res)

    ### Scatterplot functions
    def _strat_scatterplot(self, colnames, **kwargs):
        self._build_strat_plot(self._n_rows, self._n_cols, **kwargs)
        return strat_scatterplot(self._df.notHandy(), colnames[0], colnames[1])

    @inccol
    def scatterplot(self, colnames, ax=None, **kwargs):
        assert len(colnames) == 2, "There must be two columns to plot!"
        check_columns(self._df, colnames)
        colnames = [col for col in colnames if col in self._numerical]
        assert len(colnames) == 2, "Both columns must be numerical!"
        return scatterplot(self._df, colnames[0], colnames[1], ax=ax)

    ### Histogram functions
    def _strat_hist(self, colname, bins=10, **kwargs):
        self._build_strat_plot(self._n_rows, self._n_cols, **kwargs)
        categorical = True
        if colname in self._continuous:
            categorical = False
        #res = strat_histogram(self._df.notHandy(), colname, bins, categorical)
        res = strat_histogram(self._df, colname, bins, categorical)
        self._strata_plot[0].suptitle('')
        plt.tight_layout()
        return res

    @inccol
    def hist(self, colname, bins=10, ax=None, **kwargs):
        # TO DO
        # include split per response/columns
        assert len(ensure_list(colname)) == 1, "Only single columns can be plot!"
        check_columns(self._df, colname)
        if colname in self._continuous:
            return histogram(self._df, colname, bins=bins, categorical=False, ax=ax)
        else:
            return histogram(self._df, colname, bins=bins, categorical=True, ax=ax)


class HandyGrouped(GroupedData):
    def __init__(self, jgd, df, *args):
        self._jgd = jgd
        self._df = df
        self.sql_ctx = df.sql_ctx
        self._cols = args

    def agg(self, *exprs):
        df = super().agg(*exprs)
        handy = deepcopy(self._df._handy)
        handy._group_cols = self._cols
        return HandyFrame(df, handy)

    def __repr__(self):
        return "HandyGrouped[%s]" % (", ".join("%s" % c for c in self._group_cols))


class HandyFrame(DataFrame):
    """HandySpark version of DataFrame.

    Attributes
    ----------
    cols: HandyColumns
        class to access pandas-like column based methods implemented in Spark
    pandas: HandyPandas
        class to access pandas-like column based methods through pandas UDFs
    transformers: HandyTransformers
        class to generate Handy transformers
    stages: integer
        number of stages in the execution plan
    response: string
        name of the response column
    is_classification: boolean
        True if response is a categorical variable
    classes: list
        list of classes for a classification problem
    nclasses: integer
        number of classes for a classification problem
    ncols: integer
        number of columns of the HandyFrame
    nrows: integer
        number of rows of the HandyFrame
    shape: tuple
        tuple representing dimensionality of the HandyFrame
    statistics_: dict
        imputation fill value for each feature
        If stratified, first level keys are filter clauses for stratification
    fences_: dict
        fence values for each feature
        If stratified, first level keys are filter clauses for stratification
    is_stratified: boolean
        True if HandyFrame was stratified
    values: ndarray
        Numpy representation of HandyFrame.

    Available methods:
    - notHandy: makes it a plain Spark dataframe
    - stratify: used to perform stratified operations
    - isnull: checks for missing values
    - fill: fills missing values
    - outliers: returns counts of outliers, columnwise, using Tukey's method
    - get_outliers: returns list of outliers using Mahalanobis distance
    - remove_outliers: filters out outliers using Mahalanobis distance
    - fence: fences outliers
    - set_safety_limit: defines new safety limit for collect operations
    - safety_off: disables safety limit for a single operation
    - assign: appends a new columns based on an expression
    - nunique: returns number of unique values in each column
    - set_response: sets column to be used as response / label
    - disassemble: turns a vector / array column into multiple columns
    - to_metrics_RDD: turns probability and label columns into a tuple RDD
    """

    def __init__(self, df, handy=None):
        super().__init__(df._jdf, df.sql_ctx)
        if handy is None:
            handy = Handy(self)
        else:
            handy = deepcopy(handy)
            handy._df = self
            handy._update_types()
        self._handy = handy
        self._safety = self._handy._safety
        self._safety_limit = self._handy._safety_limit
        self.__overriden = ['collect', 'take']
        self._strat_handy = None
        self._strat_index = None

    def __getattribute__(self, name):
        attr = object.__getattribute__(self, name)
        if hasattr(attr, '__call__') and name not in self.__overriden:
            def wrapper(*args, **kwargs):
                try:
                    res = attr(*args, **kwargs)
                except HandyException as e:
                    raise HandyException(str(e), summary=False)
                except Exception as e:
                    raise HandyException(str(e), summary=True)

                if name != 'notHandy':
                    if not isinstance(res, HandyFrame):
                        if isinstance(res, DataFrame):
                            res = HandyFrame(res, self._handy)
                        if isinstance(res, GroupedData):
                            res = HandyGrouped(res._jgd, res._df, *args)
                return res
            return wrapper
        else:
            return attr

    def __repr__(self):
        return "HandyFrame[%s]" % (", ".join("%s: %s" % c for c in self.dtypes))

    def _get_strata(self):
        plot = None
        object = None
        if self._strat_handy is not None:
            try:
                object = self._strat_handy._strata_object
            except AttributeError:
                pass
            if object is None:
                object = True
            try:
                plots = self._strat_handy._strata_plot[1]
                #if len(plots) > 1:
                #    plot = plots[self._strat_index]
                plot = plots
            except (AttributeError, IndexError):
                pass
        return plot, object

    def _gen_row_ids(self, *args):
        # EXPERIMENTAL - DO NOT USE!
        return (self
                .sort(*args)
                .withColumn('_miid', F.monotonically_increasing_id())
                .withColumn('_row_id', F.row_number().over(Window().orderBy(F.col('_miid'))))
                .drop('_miid'))

    def _loc(self, lower_bound, upper_bound):
        # EXPERIMENTAL - DO NOT USE!
        assert '_row_id' in self.columns, "Cannot use LOC without generating `row_id`s first!"
        clause = F.col('_row_id').between(lower_bound, upper_bound)
        return self.filter(clause)

    @property
    def cols(self):
        """Returns a class to access pandas-like column based methods implemented in Spark

        Available methods:
        - min
        - max
        - median
        - q1
        - q3
        - stddev
        - value_counts
        - mode
        - corr
        - nunique
        - hist
        - boxplot
        - scatterplot
        """
        return HandyColumns(self, self._handy)

    @property
    def pandas(self):
        """Returns a class to access pandas-like column based methods through pandas UDFs

        Available methods:
        - betweeen / between_time
        - isin
        - isna / isnull
        - notna / notnull
        - abs
        - clip / clip_lower / clip_upper
        - replace
        - round / truncate
        - tz_convert / tz_localize
        """
        return HandyPandas(self)

    @property
    def transformers(self):
        """Returns a class to generate Handy transformers

        Available transformers:
        - HandyImputer
        - HandyFencer
        """
        return HandyTransformers(self)

    @property
    def stages(self):
        """Returns the number of stages in the execution plan.
        """
        return self._handy.stages

    @property
    def response(self):
        """Returns the name of the response column.
        """
        return self._handy.response

    @property
    def is_classification(self):
        """Returns True if response is a categorical variable.
        """
        return self._handy.is_classification

    @property
    def classes(self):
        """Returns list of classes for a classification problem.
        """
        return self._handy.classes

    @property
    def nclasses(self):
        """Returns the number of classes for a classification problem.
        """
        return self._handy.nclasses

    @property
    def ncols(self):
        """Returns the number of columns of the HandyFrame.
        """
        return self._handy.ncols

    @property
    def nrows(self):
        """Returns the number of rows of the HandyFrame.
        """
        return self._handy.nrows

    @property
    def shape(self):
        """Return a tuple representing the dimensionality of the HandyFrame.
        """
        return self._handy.shape

    @property
    def statistics_(self):
        """Returns dictionary with imputation fill value for each feature.
        If stratified, first level keys are filter clauses for stratification.
        """
        return self._handy.statistics_

    @property
    def fences_(self):
        """Returns dictionary with fence values for each feature.
        If stratified, first level keys are filter clauses for stratification.
        """
        return self._handy.fences_

    @property
    def values(self):
        """Numpy representation of HandyFrame.
        """
        # safety limit will kick in, unless explicitly off before
        tdf = self
        if self._safety:
            tdf = tdf.limit(self._safety_limit)
        return np.array(tdf.rdd.map(tuple).collect())

    def notHandy(self):
        """Converts HandyFrame back into Spark's DataFrame
        """
        return DataFrame(self._jdf, self.sql_ctx)

    def set_safety_limit(self, limit):
        """Sets safety limit used for ``collect`` method.
        """
        self._handy._safety_limit = limit
        self._safety_limit = limit

    def safety_off(self):
        """Disables safety limit for a single call of ``collect`` method.
        """
        self._handy._safety = False
        self._safety = False
        return self

    def collect(self):
        """Returns all the records as a list of :class:`Row`.

        By default, its output is limited by the safety limit.
        To get original `collect` behavior, call ``safety_off`` method first.
        """
        try:
            if self._safety:
                print('\nINFO: Safety is ON - returning up to {} instances.'.format(self._safety_limit))
                return super().limit(self._safety_limit).collect()
            else:
                res = super().collect()
                self._safety = True
                return res
        except HandyException as e:
            raise HandyException(str(e), summary=False)
        except Exception as e:
            raise HandyException(str(e), summary=True)

    def take(self, num):
        """Returns the first ``num`` rows as a :class:`list` of :class:`Row`.
        """
        self._handy._safety = False
        res = super().take(num)
        self._handy._safety = True
        return res

    def stratify(self, strata):
        """Stratify the HandyFrame.

        Stratified operations should be more efficient than group by operations, as they
        rely on three iterative steps, namely: filtering the underlying HandyFrame, performing
        the operation and aggregating the results.
        """
        strata = ensure_list(strata)
        check_columns(self, strata)
        return self._handy._stratify(strata)

    def transform(self, f, name=None, args=None, returnType=None):
        """INTERNAL USE
        """
        return HandyTransform.transform(self, f, name=name, args=args, returnType=returnType)

    def apply(self, f, name=None, args=None, returnType=None):
        """INTERNAL USE
        """
        return HandyTransform.apply(self, f, name=name, args=args, returnType=returnType)

    def assign(self, **kwargs):
        """Assign new columns to a HandyFrame, returning a new object (a copy)
        with all the original columns in addition to the new ones.

        Parameters
        ----------
        kwargs : keyword, value pairs
            keywords are the column names.
            If the values are callable, they are computed on the DataFrame and
            assigned to the new columns.
            If the values are not callable, (e.g. a scalar, or string),
            they are simply assigned.

        Returns
        -------
        df : HandyFrame
            A new HandyFrame with the new columns in addition to
            all the existing columns.
        """
        return HandyTransform.assign(self, **kwargs)

    @agg
    def isnull(self, ratio=False):
        """Returns array with counts of missing value for each column in the HandyFrame.

        Parameters
        ----------
        ratio: boolean, default False
            If True, returns ratios instead of absolute counts.

        Returns
        -------
        counts: Series
        """
        return self._handy.isnull(ratio)

    @agg
    def nunique(self):
        """Return Series with number of distinct observations for all columns.

        Parameters
        ----------
        exact: boolean, optional
            If True, computes exact number of unique values, otherwise uses an approximation.

        Returns
        -------
        nunique: Series
        """
        return self._handy.nunique(self.columns) #, exact)

    @inccol
    def outliers(self, ratio=False, method='tukey', **kwargs):
        """Return Series with number of outlier observations according to
         the specified method for all columns.

         Parameters
         ----------
         ratio: boolean, optional
            If True, returns proportion instead of counts.
            Default is True.
         method: string, optional
            Method used to detect outliers. Currently, only Tukey's method is supported.
            Default is tukey.

         Returns
         -------
         outliers: Series
        """
        return self._handy.outliers(self.columns, ratio=ratio, method=method, **kwargs)

    def get_outliers(self, colnames=None, critical_value=.999):
        """Returns HandyFrame containing all rows deemed as outliers using
        Mahalanobis distance and informed critical value.

        Parameters
        ----------
        colnames: list of str, optional
            List of columns to be used for computing Mahalanobis distance.
            Default includes all numerical columns
        critical_value: float, optional
            Critical value for chi-squared distribution to classify outliers
            according to Mahalanobis distance.
            Default is .999 (99.9%).
        """
        return self._handy.get_outliers(colnames, critical_value)

    def remove_outliers(self, colnames=None, critical_value=.999):
        """Returns HandyFrame containing only rows NOT deemed as outliers
        using  Mahalanobis distance and informed critical value.

        Parameters
        ----------
        colnames: list of str, optional
            List of columns to be used for computing Mahalanobis distance.
            Default includes all numerical columns
        critical_value: float, optional
            Critical value for chi-squared distribution to classify outliers
            according to Mahalanobis distance.
            Default is .999 (99.9%).
        """
        return self._handy.remove_outliers(colnames, critical_value)

    def set_response(self, colname):
        """Sets column to be used as response in supervised learning algorithms.

        Parameters
        ----------
        colname: string

        Returns
        -------
        self
        """
        check_columns(self, colname)
        return self._handy.set_response(colname)

    @inccol
    def fill(self, *args, categorical=None, continuous=None, strategy=None):
        """Fill NA/NaN values using the specified methods.

        The values used for imputation are kept in ``statistics_`` property
        and can later be used to generate a corresponding HandyImputer transformer.

        Parameters
        ----------
        categorical: 'all' or list of string, optional
            List of categorical columns.
            These columns are filled with its coresponding modes (most common values).
        continuous: 'all' or list of string, optional
            List of continuous value columns.
            By default, these columns are filled with its  corresponding means.
            If a same-sized list is provided in the ``strategy`` argument, it uses
            the corresponding straegy for each column.
        strategy: list of string, optional
            If informed, it must contain a strategy - either ``mean`` or ``median`` - for
            each one of the continuous columns.

        Returns
        -------
        df : HandyFrame
            A new HandyFrame with filled missing values.
        """
        return self._handy.fill(*args, continuous=continuous, categorical=categorical, strategy=strategy)

    @inccol
    def fence(self, colnames, k=1.5):
        """Caps outliers using lower and upper fences given by Tukey's method,
        using 1.5 times the interquartile range (IQR).

        The fence values used for capping outliers are kept in ``fences_`` property
        and can later be used to generate a corresponding HandyFencer transformer.

        For more information, check: https://en.wikipedia.org/wiki/Outlier#Tukey's_fences

        Parameters
        ----------
        colnames: list of string
            Column names to apply fencing.
        k: float, optional
            Constant multiplier for the IQR.
            Default is 1.5 (corresponding to Tukey's outlier, use 3 for "far out" values)

        Returns
        -------
        df : HandyFrame
            A new HandyFrame with capped outliers.
        """
        return self._handy.fence(colnames, k=k)

    def disassemble(self, colname, new_colnames=None):
        """Disassembles a Vector or Array column into multiple columns.

        Parameters
        ----------
        colname: string
            Column containing Vector or Array elements.
        new_colnames: list of string, optional
            Default is None, column names are generated using a sequentially
            generated suffix (e.g., _0, _1, etc.) for ``colname``.
            If informed, it must have as many column names as elements
            in the shortest vector/array of ``colname``.

        Returns
        -------
        df : HandyFrame
            A new HandyFrame with the new disassembled columns in addition to
            all the existing columns.
        """
        return self._handy.disassemble(colname, new_colnames)

    def to_metrics_RDD(self, prob_col='probability', label_col='label'):
        """Converts a DataFrame containing predicted probabilities and classification labels
        into a RDD suited for use with ``BinaryClassificationMetrics`` object.

        Parameters
        ----------
        prob_col: string, optional
            Column containing Vectors of probabilities.
            Default is 'probability'.
        label_col: string, optional
            Column containing labels.
            Default is 'label'.

        Returns
        -------
        rdd: RDD
            RDD of tuples (probability, label)
        """
        return self._handy.to_metrics_RDD(prob_col, label_col)


class Bucket(object):
    """Bucketizes a column of continuous values into equal sized bins
    to perform stratification.

    Parameters
    ----------
    colname: string
        Column containing continuous values
    bins: integer
        Number of equal sized bins to map original values to.

    Returns
    -------
    bucket: Bucket
        Bucket object to be used as column in stratification.
    """
    def __init__(self, colname, bins=5):
        self._colname = colname
        self._bins = bins
        self._buckets = None
        self._clauses = None

    def __repr__(self):
        return 'Bucket_{}_{}'.format(self._colname, self._bins)

    @property
    def colname(self):
        return self._colname

    def _get_buckets(self, df):
        check_columns(df, self._colname)
        buckets = ([-float('inf')] +
                   np.linspace(*df.agg(F.min(self._colname),
                                       F.max(self._colname)).rdd.map(tuple).collect()[0],
                               self._bins + 1).tolist() +
                   [float('inf')])
        buckets[-2] += 1e-7
        self._buckets = buckets
        return buckets

    def _get_clauses(self, buckets):
        clauses = []
        clauses.append('{} < {:.4f}'.format(self._colname, buckets[1]))
        for b, e in zip(buckets[1:-2], buckets[2:-1]):
            clauses.append('{} >= {:.4f} and {} < {:.4f}'.format(self._colname, b, self._colname, e))
        clauses[-1] = clauses[-1].replace('<', '<=')
        clauses.append('{} > {:.4f}'.format(self._colname, buckets[-2]))
        self._clauses = clauses
        return clauses


class Quantile(Bucket):
    """Bucketizes a column of continuous values into quantiles
    to perform stratification.

    Parameters
    ----------
    colname: string
        Column containing continuous values
    bins: integer
        Number of quantiles to map original values to.

    Returns
    -------
    quantile: Quantile
        Quantile object to be used as column in stratification.
    """
    def __repr__(self):
        return 'Quantile{}_{}'.format(self._colname, self._bins)

    def _get_buckets(self, df):
        buckets = ([-float('inf')] +
                   df.approxQuantile(col=self._colname,
                                     probabilities=np.linspace(0, 1, self._bins + 1).tolist(),
                                     relativeError=0.01) +
                   [float('inf')])
        buckets[-2] += 1e-7
        return buckets


class HandyColumns(object):
    """HandyColumn(s) in a HandyFrame.

    Attributes
    ----------
    numerical: list of string
        List of numerical columns (integer, float, double)
    categorical: list of string
        List of categorical columns (string, integer)
    continuous: list of string
        List of continous columns (float, double)
    string: list of string
        List of string columns (string)
    array: list of string
        List of array columns (array, map)
    """
    def __init__(self, df, handy, strata=None):
        self._df = df
        self._handy = handy
        self._strata = strata
        self._colnames = None
        self.COLTYPES = {'continuous': self.continuous,
                         'categorical': self.categorical,
                         'numerical': self.numerical,
                         'string': self.string,
                         'array': self.array}

    def __getitem__(self, *args):
        if isinstance(args[0], tuple):
            args = args[0]
        item = args[0]
        if self._strata is None:
            if self._colnames is None:
                if item == slice(None, None, None):
                    item = self._df.columns

                if isinstance(item, str):
                    try:
                        # try it as an alias
                        item = self.COLTYPES[item]
                    except KeyError:
                        pass

                check_columns(self._df, item)
                self._colnames = item

                if isinstance(self._colnames, int):
                    idx = self._colnames + (len(self._handy._group_cols) if self._handy._group_cols is not None else 0)
                    assert idx < len(self._df.columns), "Invalid column index {}".format(idx)
                    self._colnames = list(self._df.columns)[idx]

                return self
            else:
                try:
                    n = item.stop
                    if n is None:
                        n = -1
                except:
                    n = 20

                if isinstance(self._colnames, (tuple, list)):
                    res = self._df.notHandy().select(self._colnames)
                    if n == -1:
                        if self._df._safety:
                            print('\nINFO: Safety is ON - returning up to {} instances.'.format(self._df._safety_limit))
                            n = self._df._safety_limit
                    if n != -1:
                        res = res.limit(n)
                    res = res.toPandas()
                    self._handy._safety = True
                    self._df._safety = True
                    return res
                else:
                    return self._handy.__getitem__(self._colnames, n)
        else:
            if self._colnames is None:
                if item == slice(None, None, None):
                    item = self._df.columns

                if isinstance(item, str):
                    try:
                        # try it as an alias
                        item = self.COLTYPES[item]
                    except KeyError:
                        pass

            self._strata._handycolumns = item
            return self._strata

    def __repr__(self):
        colnames = ensure_list(self._colnames)
        return "HandyColumns[%s]" % (", ".join("%s" % str(c) for c in colnames))

    @property
    def numerical(self):
        """Returns list of numerical columns in the HandyFrame.
        """
        return self._handy._numerical

    @property
    def categorical(self):
        """Returns list of categorical columns in the HandyFrame.
        """
        return self._handy._categorical

    @property
    def continuous(self):
        """Returns list of continuous columns in the HandyFrame.
        """
        return self._handy._continuous

    @property
    def string(self):
        """Returns list of string columns in the HandyFrame.
        """
        return self._handy._string

    @property
    def array(self):
        """Returns list of array or map columns in the HandyFrame.
        """
        return self._handy._array

    def mean(self):
        return self._handy.mean(self._colnames)

    def min(self):
        return self._handy.min(self._colnames)

    def max(self):
        return self._handy.max(self._colnames)

    def median(self, precision=.01):
        """Returns approximate median with given precision.

        Parameters
        ----------
        precision: float, optional
            Default is 0.01
        """
        return self._handy.median(self._colnames, precision)

    def stddev(self):
        return self._handy.stddev(self._colnames)

    def var(self):
        return self._handy.var(self._colnames)

    def percentile(self, perc, precision=.01):
        """Returns approximate percentile with given precision.

        Parameters
        ----------
        perc: integer
            Percentile to be computed
        precision: float, optional
            Default is 0.01
        """
        return self._handy.percentile(self._colnames, perc, precision)

    def q1(self, precision=.01):
        """Returns approximate first quartile with given precision.

        Parameters
        ----------
        precision: float, optional
            Default is 0.01
        """
        return self._handy.q1(self._colnames, precision)

    def q3(self, precision=.01):
        """Returns approximate third quartile with given precision.

        Parameters
        ----------
        precision: float, optional
            Default is 0.01
        """
        return self._handy.q3(self._colnames, precision)

    def _value_counts(self, dropna=True, raw=True):
        assert len(ensure_list(self._colnames)) == 1, "A single column must be selected!"
        return self._handy._value_counts(self._colnames, dropna, raw)

    def value_counts(self, dropna=True):
        """Returns object containing counts of unique values.

        The resulting object will be in descending order so that the
        first element is the most frequently-occurring element.
        Excludes NA values by default.


        Parameters
        ----------
        dropna : boolean, default True
            Don't include counts of missing values.

        Returns
        -------
        counts: Series
        """
        assert len(ensure_list(self._colnames)) == 1, "A single column must be selected!"
        return self._handy.value_counts(self._colnames, dropna)

    def entropy(self):
        """Returns object containing entropy (base 2) of each column.

        Returns
        -------
        entropy: Series
        """
        return self._handy.entropy(self._colnames)

    def mutual_info(self):
        """Returns object containing matrix of mutual information
        between every pair of columns.

        Returns
        -------
        mutual_info: pd.DataFrame
        """
        return self._handy.mutual_info(self._colnames)

    def mode(self):
        """Returns same-type modal (most common) value for each column.

        Returns
        -------
        mode: Series
        """
        colnames = ensure_list(self._colnames)
        modes = [self._handy.mode(colname) for colname in colnames]
        if len(colnames) == 1:
            return modes[0]
        else:
            return pd.concat(modes, axis=0)

    def corr(self, method='pearson'):
        """Compute pairwise correlation of columns, excluding NA/null values.

        Parameters
        ----------
        method : {'pearson', 'spearman'}
            * pearson : standard correlation coefficient
            * spearman : Spearman rank correlation

        Returns
        -------
        y : DataFrame
        """
        colnames = [col for col in self._colnames if col in self.numerical]
        return self._handy.corr(colnames, method=method)

    def nunique(self):
        """Return Series with number of distinct observations for specified columns.

        Parameters
        ----------
        exact: boolean, optional
            If True, computes exact number of unique values, otherwise uses an approximation.

        Returns
        -------
        nunique: Series
        """
        return self._handy.nunique(self._colnames) #, exact)

    def outliers(self, ratio=False, method='tukey', **kwargs):
        """Return Series with number of outlier observations according to
         the specified method for all columns.

         Parameters
         ----------
         ratio: boolean, optional
            If True, returns proportion instead of counts.
            Default is True.
         method: string, optional
            Method used to detect outliers. Currently, only Tukey's method is supported.
            Default is tukey.

         Returns
         -------
         outliers: Series
        """
        return self._handy.outliers(self._colnames, ratio=ratio, method=method, **kwargs)

    def get_outliers(self, critical_value=.999):
        """Returns HandyFrame containing all rows deemed as outliers using
        Mahalanobis distance and informed critical value.

        Parameters
        ----------
        critical_value: float, optional
            Critical value for chi-squared distribution to classify outliers
            according to Mahalanobis distance.
            Default is .999 (99.9%).
        """
        return self._handy.get_outliers(self._colnames, critical_value)

    def remove_outliers(self, critical_value=.999):
        """Returns HandyFrame containing only rows NOT deemed as outliers
        using  Mahalanobis distance and informed critical value.

        Parameters
        ----------
        critical_value: float, optional
            Critical value for chi-squared distribution to classify outliers
            according to Mahalanobis distance.
            Default is .999 (99.9%).
        """
        return self._handy.remove_outliers(self._colnames, critical_value)

    def hist(self, bins=10, ax=None):
        """Draws histogram of the HandyFrame's column using matplotlib / pylab.

        Parameters
        ----------
        bins : integer, default 10
            Number of histogram bins to be used
        ax : matplotlib axes object, default None
        """
        return self._handy.hist(self._colnames, bins, ax)

    def boxplot(self, ax=None, showfliers=True, k=1.5, precision=.01):
        """Makes a box plot from HandyFrame column.

        Parameters
        ----------
        ax : matplotlib axes object, default None
        showfliers : bool, optional (True)
            Show the outliers beyond the caps.
        k: float, optional
            Constant multiplier for the IQR.
            Default is 1.5 (corresponding to Tukey's outlier, use 3 for "far out" values)
        """
        return self._handy.boxplot(self._colnames, ax, showfliers, k, precision)

    def scatterplot(self, ax=None):
        """Makes a scatter plot of two HandyFrame columns.

        Parameters
        ----------
        ax : matplotlib axes object, default None
        """
        return self._handy.scatterplot(self._colnames, ax)


class HandyStrata(object):
    __handy_methods = (list(filter(lambda n: n[0] != '_',
                               (map(itemgetter(0),
                                    inspect.getmembers(HandyFrame,
                                                       predicate=inspect.isfunction) +
                                    inspect.getmembers(HandyColumns,
                                                       predicate=inspect.isfunction)))))) + ['handy']

    def __init__(self, handy, strata):
        self._handy = handy
        self._df = handy._df
        self._strata = strata
        self._col_clauses = []
        self._colnames = []
        self._temp_colnames = []

        temp_df = self._df
        temp_df._handy = self._handy
        for col in self._strata:
            clauses = []
            colname = str(col)
            self._colnames.append(colname)
            if isinstance(col, Bucket):
                self._temp_colnames.append(colname)
                buckets = col._get_buckets(self._df)
                clauses = col._get_clauses(buckets)
                bucketizer = Bucketizer(splits=buckets, inputCol=col.colname, outputCol=colname)
                temp_df = HandyFrame(bucketizer.transform(temp_df), self._handy)
            self._col_clauses.append(clauses)

        self._df = temp_df
        self._handy._df = temp_df
        self._df._handy = self._handy

        value_counts = self._df._handy._value_counts(self._colnames, raw=True).reset_index()
        self._raw_combinations = sorted(list(map(tuple, zip(*[value_counts[colname].values
                                                              for colname in self._colnames]))))
        self._raw_clauses = [' and '.join('{} == {}'.format(str(col), value) if isinstance(col, Bucket)
                                      else  '{} == "{}"'.format(str(col),
                                                                value[0] if isinstance(value, tuple) else value)
                                      for col, value in zip(self._strata, comb))
                         for comb in self._raw_combinations]

        self._combinations = [tuple(value if not len(clauses) else clauses[int(float(value))]
                                    for value, clauses in zip(comb, self._col_clauses))
                              for comb in self._raw_combinations]
        self._clauses = [' and '.join(value if isinstance(col, Bucket)
                                      else  '{} == "{}"'.format(str(col),
                                                                value[0] if isinstance(value, tuple) else value)
                                      for col, value in zip(self._strata, comb))
                         for comb in self._combinations]
        self._strat_df = [self._df.filter(clause) for clause in self._clauses]

        self._df._strat_handy = self._handy
        # Shares the same HANDY object among all sub dataframes
        for i, df in enumerate(self._strat_df):
            df._strat_index = i
            df._strat_handy = self._handy
        self._imputed_values = {}
        self._handycolumns = None

    def __repr__(self):
        repr = "HandyStrata[%s]" % (", ".join("%s" % str(c) for c in self._strata))
        if self._handycolumns is not None:
            colnames = ensure_list(self._handycolumns)
            repr = "HandyColumns[%s] by %s" % (", ".join("%s" % str(c) for c in colnames), repr)
        return repr

    def __getattribute__(self, name):
        try:
            if name == 'cols':
                return HandyColumns(self._df, self._handy, self)
            else:
                attr = object.__getattribute__(self, name)
                return attr
        except AttributeError as e:
            if name in self.__handy_methods:
                def wrapper(*args, **kwargs):
                    raised = True
                    try:
                        # Makes stratification
                        for df in self._strat_df:
                            df._handy._strata = self._strata
                        self._handy._set_stratification(self._strata,
                                                        self._raw_combinations, self._raw_clauses,
                                                        self._combinations, self._clauses)

                        if self._handycolumns is not None:
                            args = (self._handycolumns,) + args

                        try:
                            attr_strata = getattr(self._handy, '_strat_{}'.format(name))
                            self._handy._strata_object = attr_strata(*args, **kwargs)
                        except AttributeError:
                            pass

                        try:
                            if self._handycolumns is not None:
                                f = object.__getattribute__(self._handy, name)
                            else:
                                f = object.__getattribute__(self._df, name)
                            is_agg = getattr(f, '__is_agg', False)
                            is_inccol = getattr(f, '__is_inccol', False)
                        except AttributeError:
                            is_agg = False
                            is_inccol = False

                        if is_agg or is_inccol:
                            if self._handycolumns is not None:
                                colnames = ensure_list(args[0])
                            else:
                                colnames = self._df.columns
                            res = getattr(self._handy, name)(*args, **kwargs)
                        else:
                            if self._handycolumns is not None:
                                res = [getattr(df._handy, name)(*args, **kwargs) for df in self._strat_df]
                            else:
                                res = [getattr(df, name)(*args, **kwargs) for df in self._strat_df]

                        if isinstance(res, pd.DataFrame):
                            if len(self._handy.strata_colnames):
                                res = res.set_index(self._handy.strata_colnames).sort_index()
                            if is_agg:
                                if len(colnames) == 1:
                                    res = res[colnames[0]]

                        try:
                            attr_post = getattr(self._handy, '_post_{}'.format(name))
                            res = attr_post(res)
                        except AttributeError:
                            pass

                        strata = list(map(lambda v: v[1].to_dict(OrderedDict), self._handy.strata.iterrows()))
                        strata_cols = [c if isinstance(c, str) else c.colname for c in self._strata]
                        if isinstance(res, list):
                            if isinstance(res[0], DataFrame):
                                joined_df = res[0]
                                self._imputed_values = joined_df.statistics_
                                self._fenced_values = joined_df.fences_
                                if len(res) > 1:
                                    if len(joined_df.statistics_):
                                        self._imputed_values = {self._clauses[0]: joined_df.statistics_}
                                    if len(joined_df.fences_):
                                        self._fenced_values = {self._clauses[0]: joined_df.fences_}
                                    for strat_df, clause in zip(res[1:], self._clauses[1:]):
                                        if len(joined_df.statistics_):
                                            self._imputed_values.update({clause: strat_df.statistics_})
                                        if len(joined_df.fences_):
                                            self._fenced_values.update({clause: strat_df.fences_})
                                        joined_df = joined_df.unionAll(strat_df)
                                    # Clears stratification
                                    self._handy._clear_stratification()
                                    self._df._strat_handy = None
                                    self._df._strat_index = None

                                    if len(self._temp_colnames):
                                        joined_df = joined_df.drop(*self._temp_colnames)

                                    res = HandyFrame(joined_df, self._handy)
                                    res._handy._imputed_values = self._imputed_values
                                    res._handy._fenced_values = self._fenced_values
                            elif isinstance(res[0], pd.DataFrame):
                                strat_res = []
                                indexes = res[0].index.names
                                if indexes[0] is None:
                                    indexes = ['index']
                                for r, s in zip(res, strata):
                                    strata_dict = dict([(k if isinstance(k, str) else k.colname, v) for k, v in s.items()])
                                    strat_res.append(r.assign(**strata_dict)
                                                     .reset_index())
                                res = (pd.concat(strat_res)
                                       .sort_values(by=strata_cols)
                                       .set_index(strata_cols + indexes)
                                       .sort_index())
                            elif isinstance(res[0], pd.Series):
                                # TODO: TEST
                                strat_res = []
                                for r, s in zip(res, strata):
                                    strata_dict = dict([(k if isinstance(k, str) else k.colname, v) for k, v in s.items()])
                                    series_name = none2default(r.name, 0)
                                    if series_name == name:
                                        series_name = 'index'
                                    strat_res.append(r.reset_index()
                                                     .rename(columns={series_name: name, 'index': series_name})
                                                     .assign(**strata_dict)
                                                     .set_index(strata_cols + [series_name])[name])
                                res = pd.concat(strat_res).sort_index()
                                if len(ensure_list(self._handycolumns)) > 1:
                                    try:
                                        res = res.astype(np.float64)
                                        res = res.to_frame().reset_index().pivot_table(values=name,
                                                                                       index=strata_cols,
                                                                                       columns=series_name)
                                        res.columns.name = ''
                                    except ValueError:
                                        pass
                            elif isinstance(res[0], np.ndarray):
                                # TODO: TEST
                                strat_res = []
                                for r, s in zip(res, strata):
                                    strata_dict = dict([(k if isinstance(k, str) else k.colname, v) for k, v in s.items()])
                                    strat_res.append(pd.DataFrame(r, columns=[name])
                                                     .assign(**strata_dict)
                                                     .set_index(strata_cols)[name])
                                res = pd.concat(strat_res).sort_index()
                            elif isinstance(res[0], Axes):
                                res, axs = self._handy._strata_plot
                                res = consolidate_plots(res, axs, args[0], self._clauses)
                            elif isinstance(res[0], list):
                                joined_list = res[0]
                                for l in res[1:]:
                                    joined_list += l
                                return joined_list
                            elif len(res) == len(self._combinations):
                                # TODO: TEST
                                strata_df = pd.DataFrame(strata)
                                strata_df.columns = strata_cols
                                res = (pd.concat([pd.DataFrame(res, columns=[name]), strata_df], axis=1)
                                       .set_index(strata_cols)
                                       .sort_index())
                        raised = False
                        return res
                    except HandyException as e:
                        raise HandyException(str(e), summary=False)
                    except Exception as e:
                        raise HandyException(str(e), summary=True)
                    finally:
                        if not raised:
                            if isinstance(res, HandyFrame):
                                res._handy._clear_stratification()

                        self._handy._clear_stratification()
                        self._df._strat_handy = None
                        self._df._strat_index = None

                        if len(self._temp_colnames):
                            self._df = self._df.drop(*self._temp_colnames)
                            self._handy._df = self._df
                return wrapper
            else:
                raise e
