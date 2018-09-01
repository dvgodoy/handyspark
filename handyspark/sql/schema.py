import numpy as np
import datetime
from pyspark.sql.types import StructType

_mapping = {str: 'string',
            bool: 'boolean',
            int: 'integer',
            float: 'float',
            datetime.date: 'date',
            datetime.datetime: 'timestamp',
            np.bool: 'boolean',
            np.int8: 'byte',
            np.int16: 'short',
            np.int32: 'integer',
            np.int64: 'long',
            np.float32: 'float',
            np.float64: 'double',
            np.ndarray: 'list',
            list: 'list',
            tuple: 'list',
            dict: 'map'}

def generate_schema(colnames, coltypes, nullables=None):
    assert len(colnames) == len(coltypes), "You must specify types for all columns."
    invalid_types = set(coltypes).difference(set(_mapping.keys()))
    assert len(invalid_types) == 0, "Invalid type(s) specified: {}".format(str(invalid_types))

    if nullables is None:
        nullables = [True] * len(colnames)

    fields = [{"metadata": {}, "name": name, "nullable": nullable, "type": _mapping[typ]}
              for name, typ, nullable in zip(colnames, coltypes, nullables)]
    return StructType.fromJson({"type": "struct", "fields": fields})
