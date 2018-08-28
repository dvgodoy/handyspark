from handyspark.sql.transform import HandyTransform
import unicodedata

class HandyString(object):
    __supported = {'bool': ['contains', 'startswith', 'endswith', 'match', 'isalpha', 'isnumeric', 'isalnum', 'isdigit',
                            'isdecimal', 'isspace', 'islower', 'isupper', 'istitle'],
                   'str': ['replace', 'repeat', 'join', 'pad', 'slice', 'slice_replace', 'strip', 'wrap', 'translate',
                           'get', 'decode', 'encode', 'center', 'ljust', 'rjust', 'zfill', 'lstrip', 'rstrip',
                           'normalize', 'lower', 'upper', 'title', 'capitalize', 'swapcase'],
                   'int': ['count', 'find', 'index', 'len', 'rfind', 'rindex']}
    __unsupported = ['cat', 'extract', 'extractall', 'get_dummies', 'findall', 'split', 'rsplit', 'partition',
                     'rpartition']
    __available = sorted(__supported['bool'] + __supported['str'] + __supported['int'])
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
                    if self._df.select(colname).schema.fields[0].dataType.typeName() != 'string':
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
