import json
from pyspark.ml.base import Transformer
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.ml.param import *
from pyspark.sql import functions as F

class HandyTransformers(object):
    """Generates transformers to be used in pipelines.

    Available transformers:
    imputer: Transformer
        Imputation transformer for completing missing values.
    fencer: Transformer
        Fencer transformer for capping outliers according to lower and upper fences.
    """
    def __init__(self, df):
        self._df = df
        self._handy = df._handy

    def imputer(self):
        """
        Generates a transformer to impute missing values, using values
        from the HandyFrame
        """
        return HandyImputer().setDictValues(self._df.statistics_)

    def fencer(self):
        """
        Generates a transformer to fence outliers, using statistics
        from the HandyFrame
        """
        return HandyFencer().setDictValues(self._df.fences_)


class HasDict(Params):
    """Mixin for a Dictionary parameter.
    It dumps the dictionary into a JSON string for storage and
    reloads it whenever needed.
    """
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
    """Imputation transformer for completing missing values.

    Attributes
    ----------
    statistics : dict
        The imputation fill value for each feature. If stratified, first level keys are
        filter clauses for stratification.
    """
    def _transform(self, dataset):
        # Loads dictionary with values for imputation
        fillingValues = self.getDictValues()

        items = fillingValues.items()
        target = dataset
        # Loops over columns...
        for colname, v in items:
            # If value is another dictionary, it means we're dealing with
            # stratified imputation - the key is the filering clause
            # and its value is going to be used for imputation
            if isinstance(v, dict):
                clauses = v.keys()
                whens = ' '.join(['WHEN (({clause}) AND (isnan({col}) OR isnull({col}))) THEN {quote}{filling}{quote}'
                                 .format(clause=clause, col=colname, filling=v[clause],
                                         quote='"' if isinstance(v[clause], str) else '')
                                   for clause in clauses])
            # Otherwise uses the non-stratified dictionary to fill the values
            else:
                whens = ('WHEN (isnan({col}) OR isnull({col})) THEN {quote}{filling}{quote}'
                         .format(col=colname, filling=v,
                                 quote='"' if isinstance(v, str) else ''))

            expression = F.expr('CASE {expr} ELSE {col} END'.format(expr=whens, col=colname))
            target = target.withColumn(colname, expression)

        # If it is a HandyFrame, make it a regular DataFrame
        try:
            target = target.notHandy()
        except AttributeError:
            pass
        return target

    @property
    def statistics(self):
        return self.getDictValues()


class HandyFencer(Transformer, HasDict, DefaultParamsReadable, DefaultParamsWritable):
    """Fencer transformer for capping outliers according to lower and upper fences.

    Attributes
    ----------
    fences : dict
        The fence values for each feature. If stratified, first level keys are
        filter clauses for stratification.
    """
    def _transform(self, dataset):
        # Loads dictionary with values for fencing
        fences = self.getDictValues()

        items = fences.items()
        target = dataset
        for colname, v in items:
            # If value is another dictionary, it means we're dealing with
            # stratified imputation - the key is the filering clause
            # and its value is going to be used for imputation
            if isinstance(v, dict):
                clauses = v.keys()
                whens1 = ' '.join(['WHEN ({clause}) THEN greatest({col}, {fence})'.format(clause=clause,
                                                                                          col=colname,
                                                                                          fence=v[clause][0])
                                   for clause in clauses])
                whens2 = ' '.join(['WHEN ({clause}) THEN least({col}, {fence})'.format(clause=clause,
                                                                                       col=colname,
                                                                                       fence=v[clause][1])
                                   for clause in clauses])
                expression1 = F.expr('CASE {} END'.format(whens1))
                expression2 = F.expr('CASE {} END'.format(whens2))
            # Otherwise uses the non-stratified dictionary to fill the values
            else:
                expression1 = F.expr('greatest({col}, {fence})'.format(col=colname, fence=v[0]))
                expression2 = F.expr('least({col}, {fence})'.format(col=colname, fence=v[1]))

            target = target.withColumn(colname, expression1).withColumn(colname, expression2)

        # If it is a HandyFrame, make it a regular DataFrame
        try:
            target = target.notHandy()
        except AttributeError:
            pass
        return target

    @property
    def fences(self):
        return self.getDictValues()