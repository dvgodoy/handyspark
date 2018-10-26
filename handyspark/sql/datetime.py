from handyspark.sql.transform import HandyTransform
import pandas as pd

class HandyDatetime(object):
    __supported = {'boolean': ['is_leap_year', 'is_month_end', 'is_month_start', 'is_quarter_end', 'is_quarter_start',
                            'is_year_end', 'is_year_start'],
                   'string': ['strftime', 'tz', 'weekday_name'],
                   'integer': ['day', 'dayofweek', 'dayofyear', 'days_in_month', 'daysinmonth', 'hour', 'microsecond',
                           'minute', 'month', 'nanosecond', 'quarter', 'second', 'week', 'weekday', 'weekofyear',
                           'year'],
                   'date': ['date'],
                   'timestamp': ['ceil', 'floor', 'round', 'normalize', 'time', 'tz_convert', 'tz_localize']}
    __unsupported = ['freq', 'to_period', 'to_pydatetime']
    __functions = ['strftime', 'ceil', 'floor', 'round', 'normalize', 'tz_convert', 'tz_localize']
    __available = sorted(__supported['boolean'] + __supported['string'] + __supported['integer'] + __supported['date'] +
                         __supported['timestamp'])
    __types = {n: t for t, v in __supported.items() for n in v}
    _colname = None

    def __init__(self, df, colname):
        self._df = df
        self._colname = colname
        if self._df.notHandy().select(colname).dtypes[0][1] != 'timestamp':
            raise AttributeError('Can only use .dt accessor with datetimelike values')

    def __getattribute__(self, name):
        try:
            attr = object.__getattribute__(self, name)
            return attr
        except AttributeError as e:
            if name in self.__available:
                if name in self.__functions:
                    def wrapper(*args, **kwargs):
                        return HandyTransform.gen_pandas_udf(f=lambda col: col.dt.__getattribute__(name)(**kwargs),
                                                             args=(self._colname,),
                                                             returnType=self.__types.get(name, 'string'))
                    wrapper.__doc__ = getattr(pd.Series.dt, name).__doc__
                    return wrapper
                else:
                    func = HandyTransform.gen_pandas_udf(f=lambda col: col.dt.__getattribute__(name),
                                                         args=(self._colname,),
                                                         returnType=self.__types.get(name, 'string'))
                    func.__doc__ = getattr(pd.Series.dt, name).__doc__
                    return func
            else:
                raise e
