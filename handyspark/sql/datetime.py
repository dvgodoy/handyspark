from handyspark.sql.transform import HandyTransform

class HandyDatetime(object):
    __supported = {'bool': ['is_leap_year', 'is_month_end', 'is_month_start', 'is_quarter_end', 'is_quarter_start',
                            'is_year_end', 'is_year_start'],
                   'str': ['strftime', 'tz', 'weekday_name'],
                   'int': ['day', 'dayofweek', 'dayofyear', 'days_in_month', 'daysinmonth', 'hour', 'microsecond',
                           'minute', 'month', 'nanosecond', 'quarter', 'second', 'week', 'weekday', 'weekofyear',
                           'year'],
                   'date': ['date'],
                   'timestamp': ['ceil', 'floor', 'round', 'normalize', 'time', 'to_pydatetime', 'tz_convert',
                                 'tz_localize']}
    __unsupported = ['freq', 'to_period']
    __available = sorted(__supported['bool'] + __supported['str'] + __supported['int'] + __supported['date'] +
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
                    if self._df.select(colname).schema.fields[0].dataType.typeName() != 'timestamp':
                        raise AttributeError('Can only use .dt accessor with datetimelike values')
                    try:
                        alias = kwargs.pop('alias')
                    except KeyError:
                        alias = '{}.{}'.format(colname, name)
                    return self.__generic_str_function(f=lambda col: col.dt.__getattribute__(name)(**kwargs),
                                                       colname=colname,
                                                       name=alias,
                                                       returnType=self.__types.get(name, 'str'))
                return wrapper
            else:
                raise e
