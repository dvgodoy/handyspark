import json
from pyspark.ml.base import Transformer
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.ml.param import *
from pyspark.sql import functions as F

class HandyTransformers(object):
    def __init__(self, df):
        self._df = df
        self._handy = df._handy

    def imputer(self):
        return HandyImputer().setDictValues(self._df.statistics_)

    def fencer(self):
        return HandyFencer().setDictValues(self._df.fences_)

class HasDict(Params):
    dictValues = Param(Params._dummy(), "dictValues", "Dictionary values", typeConverter=TypeConverters.toString)

    def __init__(self):
        super(HasDict, self).__init__()
        self._setDefault(dictValues='{}')

    def setDictValues(self, value):
        """
        Sets the value of :py:attr:`dictValues`.
        """
        if isinstance(value, dict):
            value = json.dumps(value).replace('\'', '"')
        return self._set(dictValues=value)

    def getDictValues(self):
        """
        Gets the value of dictValues or its default value.
        """
        values = self.getOrDefault(self.dictValues)
        return json.loads(values)

class HandyImputer(Transformer, HasDict, DefaultParamsReadable, DefaultParamsWritable):
    def _transform(self, dataset):
        fillingValues = self.getDictValues()

        joined_df = None
        fill_dict = {}
        clauses = []
        items = fillingValues.items()
        for k, v in items:
            if isinstance(v, dict):
                clauses.append(k)
                strat_df = dataset.filter(k).fillna(v)
                joined_df = strat_df if joined_df is None else joined_df.unionAll(strat_df)

        if len(clauses):
            remainder = dataset.filter('not ({})'.format(' or '.join(map(lambda v: '({})'.format(v), clauses))))
            joined_df = joined_df.unionAll(remainder)

        for k, v in items:
            if not isinstance(v, dict):
                fill_dict.update({k: v})

        if joined_df is None:
            joined_df = dataset

        return joined_df.na.fill(fill_dict).notHandy

class HandyFencer(Transformer, HasDict, DefaultParamsReadable, DefaultParamsWritable):
    @staticmethod
    def __fence(df, values):
        colname, (lfence, ufence) = list(values.items())[0]
        return (df.withColumn('__fence', F.lit(lfence))
                .withColumn(colname, F.greatest(colname, '__fence'))
                .withColumn('__fence', F.lit(ufence))
                .withColumn(colname, F.least(colname, '__fence')))

    def _transform(self, dataset):
        columns = dataset.columns
        fences = self.getDictValues()
        items = fences.items()

        joined_df = None
        clauses = []
        for k, v in items:
            if isinstance(v, dict):
                clauses.append(k)
                strat_df = HandyFencer.__fence(dataset.filter(k), v)
                joined_df = strat_df if joined_df is None else joined_df.unionAll(strat_df)

        if len(clauses):
            remainder = dataset.filter('not ({})'.format(' or '.join(map(lambda v: '({})'.format(v), clauses))))
            joined_df = joined_df.unionAll(remainder)

        if joined_df is None:
            joined_df = dataset

        for k, v in items:
            if not isinstance(v, dict):
                joined_df = HandyFencer.__fence(joined_df, {k: v})

        return joined_df.select(columns).notHandy
