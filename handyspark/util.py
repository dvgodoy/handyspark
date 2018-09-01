from math import isnan, isinf
import traceback
from pyspark.rdd import RDD
from pyspark.sql import functions as F
from pyspark.mllib.common import _java2py, _py2java

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
