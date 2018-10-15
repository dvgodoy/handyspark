from math import isnan, isinf
import pandas as pd
from pyspark.ml.linalg import DenseVector
from pyspark.rdd import RDD
from pyspark.sql import functions as F, DataFrame
from pyspark.sql.types import ArrayType, DoubleType, StructType, StructField
from pyspark.mllib.common import _java2py, _py2java
import traceback

def none2default(value, default):
    return value if value is not None else default

def none2zero(value):
    return none2default(value, 0)

def ensure_list(value):
    if isinstance(value, (list, tuple)):
        return value
    else:
        return [value]

def check_columns(df, colnames):
    if colnames is not None:
        available = df.columns
        colnames = ensure_list(colnames)
        diff = set(colnames).difference(set(available))
        assert not len(diff), "DataFrame does not have {} column(s)".format(str(list(diff))[1:-1])

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class HandyException(Exception):
    def __init__(self, *args, **kwargs):
        try:
            summary = kwargs['summary']
            if summary:
                print(HandyException.exception_summary())
        except KeyError:
            pass

    @staticmethod
    def colortext(text, color_code):
        return color_code + text + (bcolors.ENDC if text[-4:] != bcolors.ENDC else '')

    @staticmethod
    def errortext(text):
        return HandyException.colortext(HandyException.colortext(text, bcolors.FAIL), bcolors.BOLD)

    @staticmethod
    def exception_summary():
        msg = traceback.format_exc()
        try:
            top = HandyException.errortext('-' * 75 + '\nHANDY EXCEPTION SUMMARY\n')
            bottom = HandyException.errortext('-' * 75)
            info = list(filter(lambda t: len(t) and t[0] != '\t', msg.split('\n')[::-1]))
            error = HandyException.errortext('Error\t: {}'.format(info[0]))
            idx = [t.strip()[:4] for t in info].index('File')
            where = [v.strip() for v in info[idx].strip().split(',')]
            location, line, func = where[0][5:], where[1][5:], where[2][3:]
            if 'ipython-input' in location:
                location = 'IPython - In [{}]'.format(location.split('-')[2])
            if 'pyspark' in error:
                new_msg = '\n{}\n{}\n{}'.format(top, error, bottom)
            else:
                new_msg = '\n{}\nLocation: {}\nLine\t: {}\nFunction: {}\n{}\n{}'.format(top, location, line, func, error, bottom)
            return new_msg
        except Exception as e:
            return 'This is awkward... \n{}'.format(str(e))

def get_buckets(rdd, buckets):
    """
    Extracted from pyspark.rdd.RDD.histogram function
    """
    if buckets < 1:
        raise ValueError("number of buckets must be >= 1")

    # filter out non-comparable elements
    def comparable(x):
        if x is None:
            return False
        if type(x) is float and isnan(x):
            return False
        return True

    filtered = rdd.filter(comparable)

    # faster than stats()
    def minmax(a, b):
        return min(a[0], b[0]), max(a[1], b[1])
    try:
        minv, maxv = filtered.map(lambda x: (x, x)).reduce(minmax)
    except TypeError as e:
        if " empty " in str(e):
            raise ValueError("can not generate buckets from empty RDD")
        raise

    if minv == maxv or buckets == 1:
        return [minv, maxv], [filtered.count()]

    try:
        inc = (maxv - minv) / buckets
    except TypeError:
        raise TypeError("Can not generate buckets with non-number in RDD")

    if isinf(inc):
        raise ValueError("Can not generate buckets with infinite value")

    # keep them as integer if possible
    inc = int(inc)
    if inc * buckets != maxv - minv:
        inc = (maxv - minv) * 1.0 / buckets

    buckets = [i * inc + minv for i in range(buckets)]
    buckets.append(maxv)  # fix accumulated error
    return buckets

def dense_to_array(sdf, colname, new_colname):
    sql_ctx = sdf.sql_ctx
    coltype = sdf.select(colname).dtypes[0][1]
    if coltype == 'vector':
        idx = sdf.columns.index(colname)
        schema = StructType(sdf.schema.fields + [StructField(new_colname, ArrayType(DoubleType()), True)])
        res = sql_ctx.createDataFrame(sdf.rdd.map(tuple)
                                      .map(lambda t: t + (DenseVector(t[idx]).values.tolist(),)),
                                      schema=schema)
    else:
        res = sdf.withColumn(new_colname, F.col(colname))

    if isinstance(res, DataFrame):
        res = res.toHandy
    return res

def disassemble(sdf, colname, new_colnames=None):
    array_col = '_{}'.format(colname)
    coltype = sdf.select(colname).schema.fields[0].dataType.typeName()
    if coltype in ['vectorudt', 'array']:
        tdf = dense_to_array(sdf, colname, array_col)
        size = tdf.select(F.min(F.size(array_col))).take(1)[0][0]
        if new_colnames is None:
            new_colnames = ['{}_{}'.format(colname, i) for i in range(size)]
        assert len(new_colnames) == size, \
            "There must be {} column names, only {} found!".format(size, len(new_colnames))
        res = tdf.select(*sdf.columns,
                         *(F.col(array_col).getItem(i).alias(n) for i, n in zip(range(size), new_colnames)))
    else:
        if new_colnames is None:
            new_colnames = [colname]
        res = sdf.withColumn(new_colnames[0], F.col(colname))

    if isinstance(res, DataFrame):
        res = res.toHandy
    return res

def get_jvm_class(cl):
    return 'org.apache.{}.{}'.format(cl.__module__[2:], cl.__name__)

def call_scala_method(py_class, scala_method, df, *args):
    sc = df.sql_ctx._sc
    java_class = getattr(sc._jvm , get_jvm_class(py_class))
    jdf = df.select(*(F.col(col).astype('double') for col in df.columns))._jdf
    java_obj = java_class(jdf)
    args = [_py2java(sc, a) for a in args]
    java_res = getattr(java_obj, scala_method)(*args)
    res = _java2py(sc, java_res)
    if isinstance(res, RDD):
        try:
            first = res.take(1)[0]
            if isinstance(first, dict):
                first = list(first.values())[0]
                if first.startswith('scala.Tuple'):
                    serde = sc._jvm.org.apache.spark.mllib.api.python.SerDe
                    java_res = serde.fromTuple2RDD(java_res)
                    res = _java2py(sc, java_res)
        except IndexError:
            pass
    return res

def counts_to_df(value_counts, colnames, n_points):
    pdf = pd.DataFrame(value_counts
                       .to_frame('count')
                       .reset_index()
                       .apply(lambda row: dict({'count': row['count']},
                                               **dict(zip(colnames, row['index'].toArray()))),
                              axis=1)
                       .values
                       .tolist())
    pdf['count'] /= pdf['count'].sum()
    proportions = pdf['count'] / pdf['count'].min()
    factor = int(n_points / proportions.sum())
    pdf = pd.concat([pdf[colnames], (proportions * factor).astype(int)], axis=1)
    combinations = pdf.apply(lambda row: row.to_dict(), axis=1).values.tolist()
    return pd.DataFrame([dict(v) for c in combinations for v in int(c.pop('count')) * [list(c.items())]])
