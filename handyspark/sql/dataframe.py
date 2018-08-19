from copy import deepcopy
from handyspark.plot import correlations, histogram, boxplot, scatterplot
from handyspark.sql.string import HandyString
from handyspark.sql.transform import _MAPPING, HandyTransform
from handyspark.util import HandyException
import numpy as np
from operator import itemgetter, add
import pandas as pd
from pyspark.sql import DataFrame, GroupedData, functions as F

class HandyImputer(object):
    pass

class Handy(object):
    def __init__(self, df, response=None):
        self._df = df
        self._is_classification = False
        self._nclasses = None
        self._classes = None

        self._imputed_values = {}
        self._summary = None
        self._means = None
        self._medians = None
        self._counts = None
        self._group_cols = None

        self._update_types()
        self.set_response(response)

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k != '_df':
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
            item = list(self._df.columns)[item + (len(self._group_cols) if self._group_cols is not None else 0)]

        if isinstance(item, str):
            if self._group_cols is None:
                return pd.Series(self._take_array(item, n), name=item)
            else:
                pdf = self._df.select(list(self._group_cols) + [item])
                if n != -1:
                    pdf = pdf.limit(n)
                return pdf.notHandy.toPandas().set_index(list(self._group_cols))

    @property
    def stages(self):
        return (len(list(filter(lambda v: '+' == v,
                                map(lambda s: s.strip()[0],
                                    self._df.rdd.toDebugString().decode().split('\n'))))) + 1)

    @property
    def statistics_(self):
        return self._imputed_values

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

    def _update_types(self):
        self._types = list(map(lambda t: (t.name, t.dataType.typeName()), self._df.schema.fields))
        self._numerical = list(map(itemgetter(0), filter(lambda t: t[1] not in ['string', 'array', 'map'],
                                                         self._types)))
        self._double = list(map(itemgetter(0), filter(lambda t: t[1] in ['double', 'float'], self._types)))
        self._categorical = list(map(itemgetter(0), filter(lambda t: t[1] not in ['double', 'float', 'array', 'map'],
                                                           self._types)))
        self._array = list(map(itemgetter(0), filter(lambda t: t[1] in ['array', 'map'], self._types)))

    def _take_array(self, colname, n):
        datatype = self._df.select(colname).schema.fields[0].dataType.typeName()
        rdd = self._df.select(colname).rdd.map(itemgetter(0))

        if n == -1:
            data = rdd.collect()
        else:
            data = rdd.take(n)

        return np.array(data, dtype=_MAPPING.get(datatype, 'object'))

    def _summaries(self):
        # TO DO
        # include stratified
        self._summary = self._df.notHandy.summary().toPandas().set_index('summary')
        for col in self._numerical:
            self._summary[col] = self._summary[col].astype('double')

        self._means = self._summary.loc['mean', self._double]
        self._medians = self._summary.loc['50%', self._double]
        self._counts = self._summary.loc['count'].astype('double')

    def _value_counts(self, colnames):
        values = (self._df.select(colnames)
                  .rdd
                  .map(tuple)
                  .map(lambda t: (t, 1))
                  .reduceByKey(add)
                  .sortBy(itemgetter(1), ascending=False))
        return values

    def __fill_target(self, target):
        joined_df = None
        fill_dict = {}
        items = self._imputed_values.items()
        for k, v in items:
            if isinstance(v, dict):
                strat_df = target.filter(k).fillna(v)
                joined_df = strat_df if joined_df is None else joined_df.unionAll(strat_df)
            else:
                fill_dict.update({k: v})
        res = HandyFrame(joined_df.na.fill(fill_dict), self)
        return res

    def _fill_values(self, colnames, categorical, strategy):
        self._summaries()
        values = {}
        values.update(dict(self._means[map(itemgetter(0),
                                     filter(lambda t: t[1] == 'mean', zip(colnames, strategy)))]))
        values.update(dict(self._medians[map(itemgetter(0),
                                       filter(lambda t: t[1] == 'median', zip(colnames, strategy)))]))
        values.update(dict([(col, self.mode(col))
                            for col in categorical if col in self._categorical]))
        self._imputed_values = values

    def __fill_self(self, *colnames, categorical, strategy, strata=None):
        if not len(colnames):
            colnames = self._double
        if strategy is None:
            strategy = 'mean'
        if isinstance(colnames[0], (list, tuple)):
            colnames = colnames[0]
        if isinstance(strategy, (list, tuple)):
            assert len(colnames) == len(strategy), "There must be a strategy to each column."
        else:
            strategy = [strategy] * len(colnames)

        if strata is None:
            self._fill_values(colnames, categorical, strategy)
            res = HandyFrame(self._df.na.fill(self._imputed_values), self)
            return res
        else:
            combinations = self._value_counts(strata).map(itemgetter(0)).collect()
            joined_df = None
            for t in combinations:
                clause = ' and '.join('{} == "{}"'.format(n, v[0] if isinstance(v, tuple) else v)
                                      for n, v in zip(strata, t))
                strat_df = HandyFrame(self._df.filter(clause)).fill(colnames,
                                                                    categorical=categorical,
                                                                    strategy=strategy)
                self._imputed_values.update({clause: strat_df.statistics_})
                joined_df = strat_df if joined_df is None else joined_df.unionAll(strat_df)
            return HandyFrame(self._df, self)

    def disassemble(self, colname, new_colnames=None):
        size = self._df.select(F.min(F.size(colname))).take(1)[0][0]
        if new_colnames is None:
            new_colnames = ['{}_{}'.format(colname, i) for i in range(size)]
        res = self._df.select('*', *(F.col(colname).getItem(i).alias(n) for i, n in zip(range(size), new_colnames)))
        return HandyFrame(res, self)

    def fill(self, *args, **kwargs):
        if len(args) and isinstance(args[0], DataFrame):
            return self.__fill_target(args[0])
        else:
            try:
                strategy = kwargs['strategy']
            except KeyError:
                strategy = None
            try:
                categorical = kwargs['categorical']
            except KeyError:
                categorical = []
            try:
                strata = kwargs['strata']
            except KeyError:
                strata = None
            return self.__fill_self(*args, categorical=categorical, strategy=strategy, strata=strata)

    def missing_data(self, ratio=False):
        self._summaries()
        base = 1.0
        name = 'missing'
        nrows = self.nrows
        if ratio:
            base = nrows
            name += '(ratio)'
        missing = (nrows - self._counts) / base
        missing.name = name
        return missing

    def set_response(self, colname):
        if colname is not None:
            assert colname in self._df.columns, "{} not in DataFrame".format(colname)
            self._response = colname
            if colname not in self._double:
                self._is_classification = True
                self._classes = self._df.select(colname).rdd.map(itemgetter(0)).distinct().collect()
                self._nclasses = len(self._classes)
        return HandyFrame(self._df, self)

    def sample(self, fraction, strata=None, seed=None):
        # TO DO:
        # StringIndexer for categorical columns
        stratified = False
        if isinstance(strata, (list, tuple)):
            strata = list(set(strata).intersection(set(self._df.columns)))
            if not len(strata):
                strata = None
            else:
                classes = self._df.select(strata).distinct().rdd.map(lambda row: row[0:]).collect()
        else:
            if self.is_classification:
                strata = [self._response]
                classes = [(c, ) for c in self.classes]

        if isinstance(strata, (list, tuple)):
            stratified = True

        if stratified:
            # So the whole object (self) is not included in the closure
            colnames = self._df.columns
            return (self._df.rdd.map(lambda row: (tuple(row[col] for col in strata), row[0:]))
                    .sampleByKey(withReplacement=False,
                                 fractions={cl: fraction for cl in classes},
                                 seed=seed)
                    .map(itemgetter(1))
                    .toDF(colnames))
        else:
            return HandyFrame(self._df.sample(withReplacement=False, fraction=fraction, seed=seed), self)

    def value_counts(self, colname):
        values = self._value_counts(colname).collect()
        return pd.Series(map(itemgetter(1), values),
                         index=map(lambda t: t[0][0], values),
                         name=colname)

    def mode(self, colname):
        return self._value_counts(colname).filter(lambda t: t[0] is not None).take(1)[0][0][0]

    def corr(self, colnames=None):
        if colnames is None:
            colnames = self._numerical
        if not isinstance(colnames, (tuple, list)):
            colnames = [colnames]
        pdf = correlations(self._df, colnames, ax=None, plot=False)
        return pdf

    def boxplot(self, colnames, ax=None):
        if not isinstance(colnames, (tuple, list)):
            colnames = [colnames]
        return boxplot(self._df, colnames, ax)

    def scatterplot(self, col1, col2, ax=None):
        return scatterplot(self._df, col1, col2, ax=ax)

    def hist(self, colname, bins=10, ax=None):
        # TO DO
        # include split per response/columns
        if colname in self._double:
            start_values, counts = histogram(self._df, colname, bins=bins, categorical=False, ax=ax)
            return start_values, counts
        else:
            pdf = histogram(self._df, colname, bins=bins, categorical=True, ax=ax)
            return pdf


class HandyGrouped(GroupedData):
    def __init__(self, jgd, df, *args):
        self._jgd = jgd
        self._df = df
        self.sql_ctx = df.sql_ctx
        self._cols = args

    def agg(self, *exprs):
        df = super().agg(*exprs)
        handy = deepcopy(self._df.handy)
        handy._group_cols = self._cols
        return HandyFrame(df, handy)


class HandyFrame(DataFrame):
    def __init__(self, df, handy=None, safety_off=False):
        super().__init__(df._jdf, df.sql_ctx)
        if handy is None:
            handy = Handy(self)
        else:
            handy = deepcopy(handy)
            handy._df = self
            handy._update_types()
        self._handy = handy
        self._safety_off = safety_off
        self._safety = not self._safety_off
        self._safety_limit = 1000
        self.__overriden = ['collect', 'take']

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
                    if isinstance(res, DataFrame):
                        res = HandyFrame(res, self._handy, self._safety_off)
                    if isinstance(res, GroupedData):
                        res = HandyGrouped(res._jgd, res._df, *args)
                return res
            return wrapper
        else:
            return attr

    def __repr__(self):
        return "HandyFrame[%s]" % (", ".join("%s: %s" % c for c in self.dtypes))

    @property
    def handy(self):
        return self._handy

    @property
    def notHandy(self):
        return DataFrame(self._jdf, self.sql_ctx)

    @property
    def str(self):
        return HandyString(self)

    @property
    def stages(self):
        return self._handy.stages

    @property
    def response(self):
        return self._handy.response

    @property
    def is_classification(self):
        return self._handy.is_classification

    @property
    def classes(self):
        return self._handy.classes

    @property
    def nclasses(self):
        return self._handy.nclasses

    @property
    def ncols(self):
        return self._handy.ncols

    @property
    def nrows(self):
        return self._handy.nrows

    @property
    def shape(self):
        return self._handy.shape

    @property
    def statistics_(self):
        return self._handy.statistics_

    @property
    def values(self):
        # safety limit will kick in, unless explicitly off before
        tdf = self._df.select(self._numerical)
        if self._safety:
            tdf = tdf.limit(self._safety_limit)
        return np.array(tdf.rdd.map(tuple).collect())

    def set_safety_limit(self, limit):
        self._safety_limit = limit

    def safety_off(self):
        self._safety_off = True
        return self

    def collect(self):
        if self._safety:
            print('\nINFO: Safety is ON - returning up to {} instances.'.format(self._safety_limit))
            return super().limit(self._safety_limit).collect()
        else:
            self._safety = True
            return super().collect()

    def take(self, num):
        self._safety_off = True
        return super().take(num)

    def transform(self, f, name=None):
        return HandyTransform.transform(self, f, name)

    def apply(self, f, name=None):
        return HandyTransform.apply(self, f, name)

    def assign(self, **kwargs):
        return HandyTransform.assign(self, **kwargs)

    def missing_data(self, ratio=False):
        return self._handy.missing_data(ratio)

    def set_response(self, colname):
        return self._handy.set_response(colname)

    def fill(self, *args, **kwargs):
        return self._handy.fill(*args, **kwargs)

    def corr_matrix(self, colnames=None):
        return self._handy.corr(colnames)

    def hist(self, colname, bins=10, ax=None):
        return self._handy.hist(colname, bins, ax)

    def boxplot(self, colnames, ax=None):
        return self._handy.boxplot(colnames, ax)

    def scatterplot(self, col1, col2, ax=None):
        return self._handy.scatterplot(col1, col2, ax)

    def value_counts(self, colname):
        return self._handy.value_counts(colname)

    def mode(self, colname):
        return self._handy.mode(colname)
