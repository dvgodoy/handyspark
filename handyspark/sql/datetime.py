from handyspark.sql.transform import HandyTransform

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

    def __init__(self, df):
        self._df = df

    def __generic_dt_function(self, f, colname, name=None, returnType='str'):
        if name is None:
            name=colname
        return HandyTransform.transform(self._df, f, name=name, args=(colname,), returnType=returnType)

    def __getattribute__(self, name):
        try:
            attr = object.__getattribute__(self, name)
            return attr
        except AttributeError as e:
            if name in self.__available:
                def wrapper(*args, **kwargs):
                    colname = args[0]
                    if self._df.select(colname).dtypes[0][1] != 'timestamp':
                        raise AttributeError('Can only use .dt accessor with datetimelike values')
                    try:
                        alias = kwargs.pop('alias')
                    except KeyError:
                        alias = '{}.{}'.format(colname, name)

                    if name in self.__functions:
                        return self.__generic_dt_function(f=lambda col: col.dt.__getattribute__(name)(**kwargs),
                                                          colname=colname,
                                                          name=alias,
                                                          returnType=self.__types.get(name, 'str'))
                    else:
                        return self.__generic_dt_function(f=lambda col: col.dt.__getattribute__(name),
                                                          colname=colname,
                                                          name=alias,
                                                          returnType=self.__types.get(name, 'str'))
                return wrapper
            else:
                raise e
