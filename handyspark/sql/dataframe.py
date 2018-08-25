from copy import deepcopy
from handyspark.plot import correlations, histogram, boxplot, scatterplot, model_scatterplot
from handyspark.sql.string import HandyString
from handyspark.sql.transform import _MAPPING, HandyTransform
from handyspark.util import HandyException, get_buckets
import inspect
import numpy as np
from operator import itemgetter, add
import pandas as pd
from pyspark.ml.feature import Bucketizer
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
        self._strata = None
        self._strata_combinations = None
        self._model_scatterplot = None

        self._update_types()
        self.set_response(response)

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k not in ['_df', '_model_scatterplot']:
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
            if self._group_cols is None or len(self._group_cols) == 0:
                res = pd.Series(self._take_array(item, n), name=item)
                if self._strata is not None:
                    strata = list(map(lambda v: v[1].to_dict(), self.strata.iterrows()))
                    if len(strata) == len(res):
                        res = pd.concat([pd.DataFrame(strata), res], axis=1).set_index(self._strata).sort_index()
                return res
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

    @property
    def strata(self):
        if self._strata is not None:
            return pd.DataFrame(data=self._strata_combinations, columns=self._strata)

    def _stratify(self, strata):
        self._strata = strata
        return HandyStrata(self, strata)

    def _set_combinations(self, combinations):
        if self._strata is not None:
            assert len(combinations[0]) == len(self._strata)
            self._strata_combinations = combinations

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

    def __fill_self(self, *colnames, categorical, strategy):
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

        self._fill_values(colnames, categorical, strategy)
        res = HandyFrame(self._df.na.fill(self._imputed_values), self)
        return res

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
            return self.__fill_self(*args, categorical=categorical, strategy=strategy)

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

    def _strata_scatterplot(self, col1, col2, ax=None):
        self._model_scatterplot = model_scatterplot(self._df, col1, col2)

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
    def is_stratified(self):
        return self._handy._strata is not None

    @property
    def strata(self):
        if self.is_stratified:
            return self._handy.strata

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

    def stratify(self, strata):
        return self._handy._stratify(strata)

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

class Bucket(object):
    def __init__(self, colname, bins=20):
        self._colname = colname
        self._bins = bins

    def __repr__(self):
        return 'Bucket_{}_{}'.format(self._colname, self._bins)

    @property
    def colname(self):
        return self._colname

    def _get_buckets(self, df):
        buckets = ([-float('inf')] +
                   get_buckets(df.select(self._colname).rdd.map(itemgetter(0)), self._bins) +
                   [float('inf')])
        return buckets

    def _get_clauses(self, buckets):
        clauses = []
        clauses.append('{} < {:.4f}'.format(self._colname, buckets[1]))
        for b, e in zip(buckets[1:-2], buckets[2:-1]):
            clauses.append('{} >= {:.4f} and {} < {:.4f}'.format(self._colname, b, self._colname, e))
        clauses.append('{} >= {:.4f}'.format(self._colname, buckets[-2]))
        return clauses

class HandyStrata(object):
    __handy_methods = (list(filter(lambda n: n[0] != '_',
                               (map(itemgetter(0),
                                    inspect.getmembers(HandyFrame,
                                                       predicate=inspect.isfunction))))))

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

        combinations = temp_df._handy._value_counts(self._colnames).map(itemgetter(0)).collect()
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
        for df in self._strat_df:
            df._handy = self._handy
        self._handy._set_combinations(self._combinations)
        self._imputed_values = {}

    def __getattribute__(self, name):
        try:
            attr = object.__getattribute__(self, name)
            return attr
        except AttributeError as e:
            if name in self.__handy_methods:
                def wrapper(*args, **kwargs):
                    try:
                        try:
                            attr_strata = getattr(self._handy, '_strata_{}'.format(name))
                            attr_strata(*args, **kwargs)
                        except AttributeError:
                            pass
                        res = [getattr(df, name)(*args, **kwargs) for df in self._strat_df]
                    except HandyException as e:
                        raise HandyException(str(e), summary=False)
                    except Exception as e:
                        raise HandyException(str(e), summary=True)

                    strata = list(map(lambda v: v[1].to_dict(), self._handy.strata.iterrows()))
                    if isinstance(res[0], DataFrame):
                        joined_df = res[0]
                        self._imputed_values = joined_df.statistics_
                        if len(res) > 1:
                            self._imputed_values = {self._clauses[0]: joined_df.statistics_}
                            for strat_df, clause in zip(res[1:], self._clauses[1:]):
                                self._imputed_values.update({clause: strat_df.statistics_})
                                joined_df = joined_df.unionAll(strat_df)
                            res = HandyFrame(joined_df, self._handy)
                            res._handy._imputed_values = self._imputed_values
                    elif isinstance(res[0], pd.DataFrame):
                        strat_res = []
                        for r, s in zip(res, strata):
                            strat_res.append(r.assign(**s)
                                             .reset_index())
                        res = pd.concat(strat_res).sort_values(by=self._strata).set_index(self._strata)
                    elif isinstance(res[0], pd.Series):
                        strat_res = []
                        for r, s in zip(res, strata):
                            strat_res.append(r.reset_index()
                                             .rename(columns={r.name: name, 'index': r.name})
                                             .assign(**s)
                                             .set_index(self._strata + [r.name])[name])
                        res = pd.concat(strat_res).sort_index()
                    elif isinstance(res[0], np.ndarray):
                        # TO TEST
                        strat_res = []
                        for r, s in zip(res, strata):
                            strat_res.append(pd.DataFrame(r, columns=[name])
                                             .assign(**s)
                                             .set_index(self._strata)[name])
                        res = pd.concat(strat_res).sort_index()
                    elif len(res) == len(self._combinations):
                        res = (pd.concat([pd.DataFrame(res, columns=[name]),
                                          pd.DataFrame(strata, columns=self._strata)], axis=1)
                               .set_index(self._strata)
                               .sort_index())
                    return res
                return wrapper
            else:
                raise e
