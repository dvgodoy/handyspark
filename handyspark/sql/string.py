from handyspark.sql.transform import HandyTransform
import unicodedata
import pandas as pd

class HandyString(object):
    __supported = {'boolean': ['contains', 'startswith', 'endswith', 'match', 'isalpha', 'isnumeric', 'isalnum', 'isdigit',
                            'isdecimal', 'isspace', 'islower', 'isupper', 'istitle'],
                   'string': ['replace', 'repeat', 'join', 'pad', 'slice', 'slice_replace', 'strip', 'wrap', 'translate',
                           'get', 'center', 'ljust', 'rjust', 'zfill', 'lstrip', 'rstrip',
                           'normalize', 'lower', 'upper', 'title', 'capitalize', 'swapcase'],
                   'integer': ['count', 'find', 'len', 'rfind']}
    __unsupported = ['cat', 'extract', 'extractall', 'get_dummies', 'findall', 'index', 'split', 'rsplit', 'partition',
                     'rpartition', 'rindex', 'decode', 'encode']
    __available = sorted(__supported['boolean'] + __supported['string'] + __supported['integer'])
    __types = {n: t for t, v in __supported.items() for n in v}
    _colname = None

    def __init__(self, df, colname):
        self._df = df
        self._colname = colname

    @staticmethod
    def _remove_accents(input):
        return unicodedata.normalize('NFKD', input).encode('ASCII', 'ignore').decode('unicode_escape')

    def remove_accents(self):
        return HandyTransform.gen_pandas_udf(f=lambda col: col.apply(HandyString._remove_accents),
                                             args=(self._colname,),
                                             returnType='string')

    def __getattribute__(self, name):
        try:
            attr = object.__getattribute__(self, name)
            return attr
        except AttributeError as e:
            if name in self.__available:
                def wrapper(*args, **kwargs):
                    return HandyTransform.gen_pandas_udf(f=lambda col: col.str.__getattribute__(name)(**kwargs),
                                                         args=(self._colname,),
                                                         returnType=self.__types.get(name, 'string'))
                wrapper.__doc__ = getattr(pd.Series.str, name).__doc__
                return wrapper
            else:
                raise e
