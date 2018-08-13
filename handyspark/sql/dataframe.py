from copy import deepcopy
import datetime
from handyspark.plot import correlations, histogram, boxplot
from handyspark.util import exception_summary
import inspect
import numpy as np
from operator import itemgetter, add
import pandas as pd
from pyspark.sql import DataFrame, GroupedData, functions as F
import time
import unicodedata

_MAPPING = {'string': str,
            'date': datetime.date,
            'timestamp': datetime.datetime,
            'boolean': np.bool,
            'binary': np.byte,
            'byte': np.int8,
            'short': np.int16,
            'integer': np.int32,
            'long': np.int64,
            'float': np.float32,
            'double': np.float64}


class HandyException(Exception):
    pass

class HandyTransform(object):
    _mapping = dict([(v.__name__, k) for k, v in  _MAPPING.items()])
    _mapping.update({'float': 'float', 'int': 'integer'})

    @staticmethod
    def gen_pandas_udf(f, args=None, returnType=None):
        sig = inspect.signature(f)
        if args is None:
            args = tuple(sig.parameters.keys())
        else:
            assert isinstance(args, (list, tuple)), "args must be list or tuple"
        name = '{}{}'.format(f.__name__, str(args).replace("'", ""))
        if returnType is None:
            returnType = str(sig.return_annotation.__name__)
        else:
            assert returnType in HandyTransform._mapping.keys(), "invalid returnType"
        returnType = HandyTransform._mapping.get(returnType, 'double')
        @F.pandas_udf(returnType=returnType)
        def udf(*args):
            return f(*args)
        return udf(*args).alias(name)

    @staticmethod
    def gen_grouped_pandas_udf(sdf, f):
        sig = inspect.signature(f)
        args = tuple(sig.parameters.keys())
        name = '{}{}'.format(f.__name__, str(f.__code__.co_varnames).replace("'", ""))
        returnType = HandyTransform._mapping.get(str(sig.return_annotation.__name__), 'double')
        schema = sdf.select(*args).withColumn(name, F.lit(None).cast(returnType)).schema
        @F.pandas_udf(schema, F.PandasUDFType.GROUPED_MAP)
        def pudf(pdf):
            computed = pdf.apply(lambda row: f(*tuple(row[p] for p in f.__code__.co_varnames)), axis=1)
            return pdf.assign(__computed=computed).rename(columns={'__computed': name})
        return pudf

    @staticmethod
    def transform(sdf, f, name=None, args=None, returnType=None):
        if name is None:
            name = '{}{}'.format(f.__name__, str(f.__code__.co_varnames).replace("'", ""))
        return sdf.withColumn(name, HandyTransform.gen_pandas_udf(f, args, returnType))

    @staticmethod
    def apply(sdf, f):
        return sdf.select(HandyTransform.gen_pandas_udf(f))

    @staticmethod
    def assign(sdf, **kwargs):
        for c, f in kwargs.items():
            sdf = sdf.transform(f, name=c)
        return sdf

class HandyString(object):
    __available = (list(filter(lambda n: n[0] != '_',
                               (map(itemgetter(0),
                                    inspect.getmembers(pd.Series.str([]),
                                                       predicate=inspect.ismethod))))))
    def __init__(self, df):
        self._df = df

    def __generic_str_function(self, f, colname, name=None, returnType='str'):
        if name is None:
            name=colname
        return HandyTransform.transform(self._df, f, name=name, args=(colname,), returnType=returnType)

    @staticmethod
    def _remove_accents(input):
        return unicodedata.normalize('NFKD', input).encode('ASCII', 'ignore').decode('unicode_escape')

    def remove_accents(self, colname):
        return self.__generic_str_function(lambda col: col.apply(HandyString._remove_accents), colname)

    def upper(self, colname):
        return self.__generic_str_function(lambda col: col.str.upper(), colname)

    def contains(self, colname, **kwargs):
        return self.__generic_str_function(lambda col: col.str.__getattribute__('contains')(**kwargs),
                                           colname,
                                           name='{}.contains({})'.format(colname, kwargs.get('pat', '')),
                                           returnType='bool')

class HandyImputer(object):
    pass

class Handy(object):
    def __init__(self, df, response=None):
        self._df = df
        self._is_classification = False
        self._nclasses = None
        self._classes = None

        self._imputed_values = None
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
                return pdf.toPandas().set_index(list(self._group_cols))

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
        self._numerical = list(map(itemgetter(0), filter(lambda t: t[1] not in ['string'], self._types)))
        self._double = list(map(itemgetter(0), filter(lambda t: t[1] in ['double', 'float'], self._types)))
        self._categorical = list(map(itemgetter(0), filter(lambda t: t[1] not in ['double', 'float'], self._types)))

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
        self._summary = self._df.summary().toPandas().set_index('summary')
        for col in self._numerical:
            self._summary[col] = self._summary[col].astype('double')

        self._means = self._summary.loc['mean', self._double]
        self._medians = self._summary.loc['50%', self._double]
        self._counts = self._summary.loc['count'].astype('double')

    def _value_counts(self, colname):
        # TO DO
        # include stratified
        values = (self._df.select(colname).rdd
                  .map(lambda row: (itemgetter(0)(row), 1))
                  .reduceByKey(add)
                  .sortBy(itemgetter(1), ascending=False))
        return values

    def __fill_target(self, target):
        # TO DO
        # include stratified
        res = HandyFrame(target.na.fill(self._imputed_values), self)
        return res

    def _fill_values(self, colnames, categorical, strategy):
        # TO DO
        # include stratified
        #
        #joined_df = None
        #for i in range(1, 4):
        #    strat_df = sdf.filter('Pclass == {}'.format(i)).fillna({'Fare': i})
        #    joined_df = strat_df if joined_df is None else joined_df.unionAll(strat_df)

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
        # TO DO
        # include stratified
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

    def fill(self, *args, **kwargs):
        # TO DO
        # include stratified
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
        return self._df

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
            return self._df.sample(withReplacement=False, fraction=fraction, seed=seed)

    def value_counts(self, colname):
        values = self._value_counts(colname).collect()
        return pd.Series(map(itemgetter(1), values),
                         index=map(itemgetter(0), values),
                         name=colname)

    def mode(self, colname):
        # TO DO
        # include stratified
        return self._value_counts(colname).filter(lambda t: t[0] is not None).take(1)[0][0]

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
        self.__overriden = ['collect']

    def __getattribute__(self, name):
        attr = object.__getattribute__(self, name)
        if hasattr(attr, '__call__') and name not in self.__overriden:
            def wrapper(*args, **kwargs):
                try:
                    res = attr(*args, **kwargs)
                except Exception as e:
                    time.sleep(1)
                    print(exception_summary())
                    raise e

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

    def set_safety_limit(self, limit):
        self._safety_limit = limit

    def safety_off(self):
        self._safety_off = True
        return self

    def collect(self):
        if self._safety:
            # print('\nINFO: Safety is ON - returning up to 1,000 instances.')
            return super().limit(self._safety_limit).collect()
        else:
            print('\nWARNING: Safety is OFF - `collect()` will return ALL instances!')
            self._safety = True
            return super().collect()

    def transform(self, f, name=None):
        return HandyTransform.transform(self, f, name)
        #if name is None:
        #    name = '{}{}'.format(f.__name__, str(f.__code__.co_varnames).replace("'", ""))
        #return self.withColumn(name, gen_pandas_udf(f))

    def assign(self, **kwargs):
        return HandyTransform.assign(self, **kwargs)
        #tdf = self
        #for c, f in kwargs.items():
        #    tdf = tdf.transform(f, name=c)
        #return tdf

    def apply(self, f):
        return HandyTransform.apply(self, f)
        #return self.select(gen_pandas_udf(f))

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

    def value_counts(self, colname):
        return self._handy.value_counts(colname)

    def mode(self, colname):
        return self._handy.mode(colname)
