from handyspark.sql.transform import HandyTransform
from handyspark.util import check_columns
import unicodedata

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

    def __getattribute__(self, name):
        try:
            attr = object.__getattribute__(self, name)
            return attr
        except AttributeError as e:
            if name in self.__available:
                def wrapper(*args, **kwargs):
                    colname = args[0]
                    check_columns(self._df, colname)
                    if self._df.select(colname).dtypes[0][1] != 'string':
                        raise AttributeError('Can only use .str accessor with string values')
                    try:
                        alias = kwargs.pop('alias')
                    except KeyError:
                        alias = '{}.{}'.format(colname, name)
                    return self.__generic_str_function(f=lambda col: col.str.__getattribute__(name)(**kwargs),
                                                       colname=colname,
                                                       name=alias,
                                                       returnType=self.__types.get(name, 'str'))
                return wrapper
            else:
                raise e
