from copy import deepcopy
from handyspark.ml.base import HandyTransformers
from handyspark.plot import correlations, histogram, boxplot, scatterplot, strat_scatterplot, strat_histogram,\
    consolidate_plots, post_boxplot
from handyspark.sql.pandas import HandyPandas
from handyspark.sql.transform import _MAPPING, HandyTransform
from handyspark.util import HandyException, get_buckets, dense_to_array, disassemble, ensure_list, check_columns, \
    none2default
import inspect
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import numpy as np
from operator import itemgetter, add
import pandas as pd
from pyspark.ml.feature import Bucketizer
from pyspark.sql import DataFrame, GroupedData, Window, functions as F

def toHandy(self):
    """Converts Spark DataFrame into HandyFrame.
    """
    return HandyFrame(self)

def notHandy(self):
    return self

DataFrame.toHandy = toHandy
DataFrame.notHandy = notHandy

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

    def _stratify(self, strata):
        return HandyStrata(self, strata)

    def _clear_stratification(self):
        self._strata = None
        self._strata_combinations = []
        self._strata_clauses = []
        self._n_cols = 1
        self._n_rows = 1

    def _set_stratification(self, strata, combinations, clauses):
        if strata is not None:
            assert len(combinations[0]) == len(strata), "Mismatched number of combinations and strata!"
            self._strata = strata
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

    def _value_counts(self, colnames, dropna=True):
        check_columns(self._df, colnames)
        data = self._df.notHandy().select(colnames)
        if dropna:
            data = data.dropna()
        values = (data
                  .rdd
                  .map(tuple)
                  .map(lambda t: (t, 1))
                  .reduceByKey(add)
                  .sortBy(itemgetter(1), ascending=False))
        return values

    def __fill_target(self, target):
        assert isinstance(target, DataFrame), "Target must be a DataFrame"
        joined_df = None
        fill_dict = {}
        clauses = []
        items = self._imputed_values.items()
        for k, v in items:
            if isinstance(v, dict):
                clauses.append(k)
                strat_df = target.filter(k).fillna(v)
                joined_df = strat_df if joined_df is None else joined_df.unionAll(strat_df)

        if len(clauses):
            remainder = target.filter('not ({})'.format(' or '.join(map(lambda v: '({})'.format(v), clauses))))
            joined_df = joined_df.unionAll(remainder)

        for k, v in items:
            if not isinstance(v, dict):
                fill_dict.update({k: v})

        if joined_df is None:
            joined_df = target

        res = HandyFrame(joined_df.na.fill(fill_dict), self)
        return res

    def _fill_values(self, continuous, categorical, strategy):
        values = {}
        values.update(dict(self._df._means[map(itemgetter(0),
                                     filter(lambda t: t[1] == 'mean', zip(continuous, strategy)))]))
        values.update(dict(self._df._medians[map(itemgetter(0),
                                       filter(lambda t: t[1] == 'median', zip(continuous, strategy)))]))
        values.update(dict([(col, self.mode(col).values[0])
                            for col in categorical if col in self._categorical]))
        return values

    def __fill_self(self, continuous, categorical, strategy):
        continuous = none2default(continuous, [])
        categorical = none2default(categorical, [])
        check_columns(self._df, continuous + categorical)

        strategy = none2default(strategy, 'mean')

        if continuous == 'all':
            continuous = self._continuous
        if categorical == 'all':
            categorical = self._categorical

        if isinstance(strategy, (list, tuple)):
            assert len(continuous) == len(strategy), "There must be a strategy to each column."
        else:
            strategy = [strategy] * len(continuous)

        values = self._fill_values(continuous, categorical, strategy)
        self._imputed_values.update(values)
        res = HandyFrame(self._df.notHandy().na.fill(values), self)
        return res

    def _dense_to_array(self, colname, array_colname):
        check_columns(self._df, colname)
        res = dense_to_array(self._df.notHandy(), colname, array_colname)
        return HandyFrame(res, self)

    def disassemble(self, colname, new_colnames=None):
        check_columns(self._df, colname)
        res = disassemble(self._df.notHandy(), colname, new_colnames)
        return HandyFrame(res, self)

    def to_metrics_RDD(self, prob_col, label):
        check_columns(self._df, [prob_col, label])
        return self.disassemble(prob_col).select('{}_1'.format(prob_col), F.col(label).cast('double')).rdd.map(tuple)

    def fill(self, *args, continuous=None, categorical=None, strategy=None):
        if len(args) and isinstance(args[0], DataFrame):
            return self.__fill_target(args[0])
        else:
            return self.__fill_self(continuous=continuous, categorical=categorical, strategy=strategy)

    def isnull(self, ratio=False):
        name = 'missing'
        nrows = self.nrows
        missing = (nrows - self._df._counts)
        if ratio:
            base = nrows
            name += '(ratio)'
            missing /= base
        missing.name = name
        return missing

    def outliers(self, colnames=None, ratio=False, method='tukey', **kwargs):
        colnames = none2default(colnames, self._numerical)
        colnames = ensure_list(colnames)
        check_columns(self._df, colnames)
        colnames = [col for col in colnames if col in self._numerical]

        if method == 'tukey':
            outliers = []
            try:
                k = kwargs['k']
            except KeyError:
                k = 1.5
            for colname in colnames:
                q1, q3 = self._df._summary.loc['25%', colname], self._df._summary.loc['75%', colname]
                iqr = q3 - q1
                lfence = q1 - (k * iqr)
                ufence = q3 + (k * iqr)
                outliers.append(self._df.filter(~F.col(colname).between(lfence, ufence)).count())
                if ratio:
                    outliers[-1] /= self._df._counts[colname]
            res = pd.Series(outliers, index=colnames, dtype=np.float64)

        return res

    def nunique(self, colnames=None):
        colnames = none2default(colnames, self._df.columns)
        colnames = ensure_list(colnames)
        check_columns(self._df, colnames)
        return pd.Series([self._df.notHandy().select(col).dropna().distinct().count() for col in colnames],
                         index=colnames)

    def fence(self, colnames, k=1.5):
        colnames = ensure_list(colnames)
        check_columns(self._df, colnames)
        colnames = [col for col in colnames if col in self._numerical]
        df = self._df.notHandy()
        for colname in colnames:
            q1, q3 = self._df.approxQuantile(col=colname, probabilities=[.25, .75], relativeError=0.01)
            iqr = q3 - q1
            lfence = q1 - (k * iqr)
            ufence = q3 + (k * iqr)
            self._fenced_values.update({colname: [lfence, ufence]})
            df = (df
                  .withColumn('__fence', F.lit(lfence))
                  .withColumn(colname, F.greatest(colname, '__fence'))
                  .withColumn('__fence', F.lit(ufence))
                  .withColumn(colname, F.least(colname, '__fence')))
        return HandyFrame(df.select(self._df.columns), self)

    def set_response(self, colname):
        check_columns(self._df, colname)
        self._response = colname
        if colname is not None:
            if colname not in self._continuous:
                self._is_classification = True
                self._classes = self._df.notHandy().select(colname).rdd.map(itemgetter(0)).distinct().collect()
                self._nclasses = len(self._classes)

        return self

    def value_counts(self, colname, dropna=True):
        values = self._value_counts(colname, dropna).collect()
        return pd.Series(map(itemgetter(1), values),
                         index=map(lambda t: t[0][0], values),
                         name=colname)

    def mode(self, colname):
        return pd.Series(self._value_counts(colname).filter(lambda t: t[0] is not None).take(1)[0][0][0],
                         index=[colname],
                         name='mode')

    def corr(self, colnames=None, method='pearson'):
        colnames = none2default(colnames, self._numerical)
        colnames = ensure_list(colnames)
        check_columns(self._df, colnames)
        colnames = [col for col in colnames if col in self._numerical]
        if self._strata is not None:
            colnames = sorted([col for col in colnames if col not in self._strata])
        pdf = correlations(self._df.notHandy(), colnames, method=method, ax=None, plot=False)
        return pdf

    def mean(self, colnames):
        return self._df._get_summary(colnames, 'mean').dropna()

    def min(self, colnames):
        return self._df._get_summary(colnames, 'min').dropna()

    def max(self, colnames):
        return self._df._get_summary(colnames, 'max').dropna()

    def median(self, colnames):
        return self._df._get_summary(colnames, '50%').dropna()

    def stddev(self, colnames):
        return self._df._get_summary(colnames, 'stddev').dropna()

    def var(self, colnames):
        return self._df._get_summary(colnames, 'stddev').dropna() ** 2

    def q1(self, colnames):
        return self._df._get_summary(colnames, '25%').dropna()

    def q3(self, colnames):
        return self._df._get_summary(colnames, '75%').dropna()

    ### Boxplot functions
    def _strat_boxplot(self, colnames, **kwargs):
        n_rows = n_cols = 1
        kwds = deepcopy(kwargs)
        try:
            del kwds['showfliers']
        except KeyError:
            pass
        if isinstance(colnames, (tuple, list)) and (len(colnames) > 1):
            n_rows = self._n_rows
            n_cols = self._n_cols
        self._build_strat_plot(n_rows, n_cols, **kwds)
        return None

    def boxplot(self, colnames, ax=None, showfliers=True, k=1.5, **kwargs):
        colnames = ensure_list(colnames)
        check_columns(self._df, colnames)
        colnames = [col for col in colnames if col in self._numerical]
        assert len(colnames), "Only numerical columns can be plot!"
        return boxplot(self._df, colnames, ax, showfliers, k)

    def _post_boxplot(self, res):
        return post_boxplot(self._strata_plot[1], res, self._strata_clauses)

    ### Scatterplot functions
    def _strat_scatterplot(self, colnames, **kwargs):
        self._build_strat_plot(self._n_rows, self._n_cols, **kwargs)
        return strat_scatterplot(self._df.notHandy(), colnames[0], colnames[1])

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
        res = strat_histogram(self._df.notHandy(), colname, bins, categorical)
        self._strata_plot[0].suptitle('')
        plt.tight_layout()
        return res

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
    - outliers: checks for outliers
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

        # statistics
        self._summary = None
        self._means = None
        self._medians = None
        self._counts = None
        self._summaries()

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
                if len(plots) > 1:
                    plot = plots[self._strat_index]
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

    def _summaries(self):
        self._summary = self.notHandy().summary().toPandas().set_index('summary')
        for col in self._handy._numerical:
            self._summary[col] = self._summary[col].astype('double')

        self._means = self._summary.loc['mean', self._handy._continuous]
        self._medians = self._summary.loc['50%', self._handy._continuous]
        self._counts = self._summary.loc['count'].astype('double')

    def _get_summary(self, colnames, statistic):
        colnames = ensure_list(colnames)
        colnames = [col for col in colnames if col in self._handy._numerical]
        check_columns(self, colnames)
        return self._summary.loc[statistic, colnames]

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

    def nunique(self):
        """Return Series with number of distinct observations for all columns.

        Returns
        -------
        nunique: Series
        """
        return self._handy.nunique(self.columns)

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

    def __repr__(self):
        return 'Bucket_{}_{}'.format(self._colname, self._bins)

    @property
    def colname(self):
        return self._colname

    def _get_buckets(self, df):
        check_columns(df, self._colname)
        buckets = ([-float('inf')] +
                   get_buckets(df.select(self._colname).rdd.map(itemgetter(0)), self._bins) +
                   [float('inf')])
        buckets[-2] += 1e-14
        return buckets

    def _get_clauses(self, buckets):
        clauses = []
        clauses.append('{} < {:.4f}'.format(self._colname, buckets[1]))
        for b, e in zip(buckets[1:-2], buckets[2:-1]):
            clauses.append('{} >= {:.4f} and {} < {:.4f}'.format(self._colname, b, self._colname, e))
        clauses[-1] = clauses[-1].replace('<', '<=')
        clauses.append('{} > {:.4f}'.format(self._colname, buckets[-2]))
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
        buckets[-2] += 1e-14
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
        return self._df._get_summary(self._colnames, 'mean').dropna()

    def min(self):
        return self._df._get_summary(self._colnames, 'min').dropna()

    def max(self):
        return self._df._get_summary(self._colnames, 'max').dropna()

    def median(self):
        return self._df._get_summary(self._colnames, '50%').dropna()

    def stddev(self):
        return self._df._get_summary(self._colnames, 'stddev').dropna()

    def var(self):
        return self._df._get_summary(self._colnames, 'stddev').dropna() ** 2

    def q1(self):
        return self._df._get_summary(self._colnames, '25%').dropna()

    def q3(self):
        return self._df._get_summary(self._colnames, '75%').dropna()

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

        Returns
        -------
        nunique: Series
        """
        return self._handy.nunique(self._colnames)

    def outliers(self, ratio=False, method='tukey'):
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
        return self._handy.outliers(self._colnames, ratio=ratio, method=method)

    def hist(self, bins=10, ax=None):
        """Draws histogram of the HandyFrame's column using matplotlib / pylab.

        Parameters
        ----------
        bins : integer, default 10
            Number of histogram bins to be used
        ax : matplotlib axes object, default None
        """
        return self._handy.hist(self._colnames, bins, ax)

    def boxplot(self, ax=None, showfliers=True, k=1.5):
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
        return self._handy.boxplot(self._colnames, ax, showfliers, k)

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

        temp_df = self._df
        for col in self._strata:
            clauses = []
            colname = str(col)
            self._colnames.append(colname)
            if isinstance(col, Bucket):
                buckets = col._get_buckets(self._df)
                clauses = col._get_clauses(buckets)
                bucketizer = Bucketizer(splits=buckets, inputCol=col.colname, outputCol=colname)
                temp_df = HandyFrame(bucketizer.transform(temp_df), self._handy)
            self._col_clauses.append(clauses)

        combinations = sorted(temp_df._handy._value_counts(self._colnames).map(itemgetter(0)).collect())
        self._combinations = [tuple(value if not len(clauses) else clauses[int(value)]
                                    for value, clauses in zip(comb, self._col_clauses))
                              for comb in combinations]
        self._clauses = [' and '.join(value if isinstance(col, Bucket)
                                      else  '{} == "{}"'.format(str(col),
                                                                value[0] if isinstance(value, tuple) else value)
                                      for col, value in zip(self._strata, comb))
                         for comb in self._combinations]
        self._strat_df = [self._df.filter(clause) for clause in self._clauses]

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
                    try:
                        # Makes stratification
                        for df in self._strat_df:
                            df._handy._strata = self._strata
                        self._handy._set_stratification(self._strata, self._combinations, self._clauses)

                        if self._handycolumns is not None:
                            args = (self._handycolumns,) + args

                        try:
                            attr_strata = getattr(self._handy, '_strat_{}'.format(name))
                            self._handy._strata_object = attr_strata(*args, **kwargs)
                        except AttributeError:
                            pass

                        if self._handycolumns is not None:
                            res = [getattr(df._handy, name)(*args, **kwargs) for df in self._strat_df]
                        else:
                            res = [getattr(df, name)(*args, **kwargs) for df in self._strat_df]

                        try:
                            attr_post = getattr(self._handy, '_post_{}'.format(name))
                            res = attr_post(res)
                        except AttributeError:
                            pass

                        strata = list(map(lambda v: v[1].to_dict(), self._handy.strata.iterrows()))
                        strata_cols = [c if isinstance(c, str) else c.colname for c in self._strata]
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
                            # TO TEST
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
                            res = (pd.concat([pd.DataFrame(res, columns=[name]),
                                              pd.DataFrame(strata, columns=strata_cols)], axis=1)
                                   .set_index(strata_cols)
                                   .sort_index())
                        return res
                    except HandyException as e:
                        raise HandyException(str(e), summary=False)
                    except Exception as e:
                        raise HandyException(str(e), summary=True)
                    finally:
                        self._handy._clear_stratification()
                return wrapper
            else:
                raise e
