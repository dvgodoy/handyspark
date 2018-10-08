from handyspark.sql.datetime import HandyDatetime
from handyspark.sql.string import HandyString
from handyspark.sql.transform import HandyTransform, _MAPPING

class HandyPandas(object):
    __supported = {'boolean': ['between', 'between_time', 'isin', 'isna', 'isnull', 'notna', 'notnull'],
                   'same': ['abs', 'clip', 'clip_lower', 'clip_upper', 'replace', 'round', 'truncate',
                            'tz_convert', 'tz_localize']}
    __as_series = ['rank', 'interpolate', 'pct_change', 'bfill', 'cummax', 'cummin', 'cumprod', 'cumsum', 'diff',
                   'ffill', 'fillna', 'shift']
    __available = sorted(__supported['boolean'] + __supported['same'])
    __types = {n: t for t, v in __supported.items() for n in v}

    def __init__(self, df):
        self._df = df

    @property
    def str(self):
        return HandyString(self._df)

    @property
    def dt(self):
        return HandyDatetime(self._df)

    def __generic_pandas_function(self, f, colname, name=None, returnType='str'):
        if name is None:
            name=colname
        if returnType == 'same':
            returnType = self._df.select(colname).dtypes[0][1]
        return HandyTransform.transform(self._df, f, name=name, args=(colname,), returnType=returnType)

    def __getattribute__(self, name):
        try:
            attr = object.__getattribute__(self, name)
            return attr
        except AttributeError as e:
            if name in self.__available:
                def wrapper(*args, **kwargs):
                    colname = args[0]
                    try:
                        alias = kwargs.pop('alias')
                    except KeyError:
                        alias = '{}.{}'.format(colname, name)
                    return self.__generic_pandas_function(f=lambda col: col.__getattribute__(name)(**kwargs),
                                                          colname=colname,
                                                          name=alias,
                                                          returnType=self.__types.get(name, 'str'))
                return wrapper
            else:
                raise e
