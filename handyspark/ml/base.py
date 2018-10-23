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

        joined_df = None
        fill_dict = {}
        clauses = []
        items = fillingValues.items()
        # Loops over items...
        for k, v in items:
            # If value is another dictionary, it means we're dealing with
            # stratified imputation - the key is the filering clause
            # and the value is the dictionary {column: value}
            if isinstance(v, dict):
                clauses.append(k)
                # Filters dataset according to clause and fills missing values
                strat_df = dataset.filter(k).fillna(v)
                # Rejoins the filtered datasets back together
                joined_df = strat_df if joined_df is None else joined_df.unionAll(strat_df)

        # It could happen that not all rows were handled - unseen values, for instance
        # So, the remainder rows are also rejoined to the resulting DataFrame
        if len(clauses):
            remainder = dataset.filter('not ({})'.format(' or '.join(map(lambda v: '({})'.format(v), clauses))))
            joined_df = joined_df.unionAll(remainder)

        # Time to check all items that are NOT stratified and build a dictionary for them
        for k, v in items:
            if not isinstance(v, dict):
                fill_dict.update({k: v})

        # If there was no stratified filling, assumes the original dataset
        if joined_df is None:
            joined_df = dataset

        # Finally, uses the non-stratified dictionary to fill remaining values
        res = joined_df.na.fill(fill_dict)

        # If it is a HandyFrame, make it a regular DataFrame
        try:
            res = res.notHandy()
        except AttributeError:
            pass
        return res

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
    @staticmethod
    def __fence(df, values):
        colname, (lfence, ufence) = list(values.items())[0]
        # Generates two columns, for lower and upper fences
        # and then applies `greatest` and `least` functions
        # to effectively fence the values.
        return (df.withColumn('__fence', F.lit(lfence))
                .withColumn(colname, F.greatest(colname, '__fence'))
                .withColumn('__fence', F.lit(ufence))
                .withColumn(colname, F.least(colname, '__fence'))
                .drop('__fence'))

    def _transform(self, dataset):
        columns = dataset.columns
        # Loads dictionary with values for fencing
        fences = self.getDictValues()
        items = fences.items()

        joined_df = None
        clauses = []
        # Loops over items...
        for k, v in items:
            # If value is another dictionary, it means we're dealing with
            # stratified imputation - the key is the filering clause
            # and the value is the dictionary {column: value}
            if isinstance(v, dict):
                clauses.append(k)
                # Filters dataset according to clause and applies fencing
                strat_df = HandyFencer.__fence(dataset.filter(k), v)
                # Rejoins the filtered datasets back together
                joined_df = strat_df if joined_df is None else joined_df.unionAll(strat_df)

        # It could happen that not all rows were handled - unseen values, for instance
        # So, the remainder rows are also rejoined to the resulting DataFrame
        if len(clauses):
            remainder = dataset.filter('not ({})'.format(' or '.join(map(lambda v: '({})'.format(v), clauses))))
            joined_df = joined_df.unionAll(remainder)

        # If there was no stratified filling, assumes the original dataset
        if joined_df is None:
            joined_df = dataset

        # Time to check all items that are NOT stratified and apply fencing to them
        for k, v in items:
            if not isinstance(v, dict):
                joined_df = HandyFencer.__fence(joined_df, {k: v})

        res = joined_df.select(columns)
        # If it is a HandyFrame, make it a regular DataFrame
        try:
            res = res.notHandy()
        except AttributeError:
            pass
        return res

    @property
    def fences(self):
        return self.getDictValues()