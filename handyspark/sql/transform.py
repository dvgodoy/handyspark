import datetime
import inspect
import numpy as np
from pyspark.sql import functions as F

_MAPPING = {'string': str,
            'date': datetime.date,
            'timestamp': datetime.datetime,
            'boolean': np.bool,
            'binary': np.byte,
            'byte': np.int8,
            'short': np.int16,
            'integer': np.int32,
            'long': np.int64,
            'float': np.float32,
            'double': np.float64,
            'array': np.ndarray,
            'map': dict}

class HandyTransform(object):
    _mapping = dict([(v.__name__, k) for k, v in  _MAPPING.items()])
    _mapping.update({'float': 'double', 'int': 'integer', 'list': 'array', 'bool': 'boolean'})

    @staticmethod
    def gen_pandas_udf(f, args=None, returnType=None):
        sig = inspect.signature(f)

        if args is None:
            args = tuple(sig.parameters.keys())
        assert isinstance(args, (list, tuple)), "args must be list or tuple"
        name = '{}{}'.format(f.__name__, str(args).replace("'", ""))

        signatureType = str(sig.return_annotation)[7:]
        if signatureType != '_empty':
            returnType = signatureType
            types = returnType.replace(']', '').replace('[', ',').split(',')[:3]
            for returnType in types:
                assert returnType.lower().strip() in HandyTransform._mapping.keys(), "invalid returnType"
            types = list(map(lambda t: HandyTransform._mapping[t.lower().strip()], types))
            returnType = types[0]
            if len(types) > 1:
                returnType = '<'.join([returnType, ','.join(types[1:])])
                returnType += '>'

        @F.pandas_udf(returnType=returnType)
        def udf(*args):
            return f(*args)

        return udf(*args).alias(name)

    @staticmethod
    def gen_grouped_pandas_udf(sdf, f, args=None, returnType=None):
        # TO TEST
        sig = inspect.signature(f)

        if args is None:
            args = tuple(sig.parameters.keys())
        assert isinstance(args, (list, tuple)), "args must be list or tuple"
        name = '{}{}'.format(f.__name__, str(f.__code__.co_varnames).replace("'", ""))

        signatureType = str(sig.return_annotation)[7:]
        if signatureType != '_empty':
            returnType = signatureType
            types = returnType.replace(']', '').replace('[', ',').split(',')[:3]
            for returnType in types:
                assert returnType.lower().strip() in HandyTransform._mapping.keys(), "invalid returnType"
            types = list(map(lambda t: HandyTransform._mapping[t.lower().strip()], types))
            returnType = types[0]
            if len(types) > 1:
                returnType = '<'.join([returnType, ','.join(types[1:])])
                returnType += '>'
        schema = sdf.select(*args).withColumn(name, F.lit(None).cast(returnType)).schema

        @F.pandas_udf(schema, F.PandasUDFType.GROUPED_MAP)
        def pudf(pdf):
            computed = pdf.apply(lambda row: f(*tuple(row[p] for p in f.__code__.co_varnames)), axis=1)
            return pdf.assign(__computed=computed).rename(columns={'__computed': name})

        return pudf

    @staticmethod
    def transform(sdf, f, name=None, args=None, returnType=None):
        if name is None:
            name = '{}{}'.format(f.__name__, str(f.__code__.co_varnames).replace("'", ""))
        return sdf.withColumn(name, HandyTransform.gen_pandas_udf(f, args, returnType))

    @staticmethod
    def apply(sdf, f, name=None, args=None, returnType=None):
        if name is None:
            name = '{}{}'.format(f.__name__, str(f.__code__.co_varnames).replace("'", ""))
        return sdf.select(HandyTransform.gen_pandas_udf(f, args, returnType).alias(name))

    @staticmethod
    def assign(sdf, **kwargs):
        for c, f in kwargs.items():
            typename = None
            if isinstance(f, tuple):
                f, typename = f
            if callable(f):
                args = f.__code__.co_varnames
                if len(args):
                    if typename is None:
                        typename = sdf.select(args[0]).schema.fields[0].dataType.typeName()
                    # CHANGE
                    # returnType =  _MAPPING.get(typename, str).__name__
                    sdf = sdf.transform(f, name=c, returnType=typename)
                else:
                    sdf = sdf.withColumn(c, F.lit(f()))
            else:
                sdf = sdf.withColumn(c, F.lit(f))
        return sdf

