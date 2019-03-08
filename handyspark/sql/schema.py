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

def generate_schema(columns, nullable_columns='all'):
    """
    Parameters
    ----------
    columns: dict of column names (keys) and types (values)
    nullables: list of nullable columns, optional, default is 'all'

    Returns
    -------
    schema: StructType
        Spark DataFrame schema corresponding to Python/numpy types.
    """
    columns = sorted(columns.items())
    colnames = list(map(itemgetter(0), columns))
    coltypes = list(map(itemgetter(1), columns))

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

    if nullable_columns == 'all':
        nullables = [True] * len(colnames)
    else:
        nullables = [col in nullable_columns for col in colnames]

    fields = [{"metadata": {}, "name": name, "nullable": nullable, "type": _mapping[typ]}
              for name, typ, nullable in zip(colnames, new_types, nullables)]
    return StructType.fromJson({"type": "struct", "fields": fields})
