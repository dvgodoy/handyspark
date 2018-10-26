from handyspark.sql.datetime import HandyDatetime
from handyspark.sql.string import HandyString
from handyspark.sql.transform import HandyTransform
from handyspark.util import check_columns
import pandas as pd

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
        self._colname = None

    def __getitem__(self, *args):
        if isinstance(args[0], tuple):
            args = args[0]
        item = args[0]
        check_columns(self._df, item)
        self._colname = item
        return self

    @property
    def str(self):
        """Returns a class to access pandas-like string column based methods through pandas UDFs

        Available methods:
        - contains
        - startswith / endswitch
        - match
        - isalpha / isnumeric / isalnum / isdigit / isdecimal / isspace
        - islower / isupper / istitle
        - replace
        - repeat
        - join
        - pad
        - slice / slice_replace
        - strip / lstrip / rstrip
        - wrap / center / ljust / rjust
        - translate
        - get
        - normalize
        - lower / upper / capitalize / swapcase / title
        - zfill
        - count
        - find / rfind
        - len
        """
        return HandyString(self._df, self._colname)

    @property
    def dt(self):
        """Returns a class to access pandas-like datetime column based methods through pandas UDFs

        Available methods:
        - is_leap_year / is_month_end / is_month_start / is_quarter_end / is_quarter_start / is_year_end / is_year_start
        - strftime
        - tz / time / tz_convert / tz_localize
        - day / dayofweek / dayofyear / days_in_month / daysinmonth
        - hour / microsecond / minute / nanosecond / second
        - week / weekday / weekday_name
        - month / quarter / year / weekofyear
        - date
        - ceil / floor / round
        - normalize
        """
        return HandyDatetime(self._df, self._colname)

    def __getattribute__(self, name):
        try:
            attr = object.__getattribute__(self, name)
            return attr
        except AttributeError as e:
            if name in self.__available:
                def wrapper(*args, **kwargs):
                    returnType=self.__types.get(name, 'string')
                    if returnType == 'same':
                        returnType = self._df.notHandy().select(self._colname).dtypes[0][1]
                    return HandyTransform.gen_pandas_udf(f=lambda col: col.__getattribute__(name)(**kwargs),
                                                         args=(self._colname,),
                                                         returnType=returnType)
                if name not in ['str', 'dt']:
                    wrapper.__doc__ = getattr(pd.Series, name).__doc__
                return wrapper
            else:
                raise e
