import numpy as np
import datetime
from operator import itemgetter
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
            np.ndarray: 'array',
            object: 'string',
            list: 'array',
            tuple: 'array',
            dict: 'map'}

def generate_schema(colnames, coltypes, nullables=None):
    """
    Parameters
    ----------
    colnames: list of string
    coltypes: list of type
    nullables: list of boolean, optional

    Returns
    -------
    schema: StructType
        Spark DataFrame schema corresponding to Python/numpy types.
    """
    assert len(colnames) == len(coltypes), "You must specify types for all columns."
    invalid_types = []
    new_types = []
    keys = list(map(itemgetter(0), list(_mapping.items())))
    for coltype in coltypes:
        if coltype not in keys:
            invalid_types.append(coltype)
        else:
            if coltype == np.dtype('O'):
                new_types.append(str)
            else:
                new_types.append(keys[keys.index(coltype)])
    assert len(invalid_types) == 0, "Invalid type(s) specified: {}".format(str(invalid_types))

    if nullables is None:
        nullables = [True] * len(colnames)

    fields = [{"metadata": {}, "name": name, "nullable": nullable, "type": _mapping[typ]}
              for name, typ, nullable in zip(colnames, new_types, nullables)]
    return StructType.fromJson({"type": "struct", "fields": fields})
