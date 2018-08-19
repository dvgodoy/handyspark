from math import isnan, isinf
import traceback
from pyspark.sql import Window, functions as F

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

### LOC
def loc(sdf, rows):
    return (sdf
            .withColumn('_miid', F.monotonically_increasing_id())
            .withColumn('_row_id', F.row_number().over(Window().orderBy(F.col('_miid'))))
            .filter(F.col('_row_id').between(*rows))
            .drop('_miid', '_row_id'))

# Forward fill
#from pyspark.sql import Window
#window = Window.orderBy('time').rowsBetween(Window.unboundedPreceding, Window.currentRow)
#sdf = sdf.withColumn(col,  F.last(sdf[col], ignorenulls=True).over(window))

#>>> from pyspark.statcounter import StatCounter
#>>> rdd = sc.parallelize([9, -1, 0, 99, 0, -10])
#>>> stats = rdd.aggregate(StatCounter(), StatCounter.merge, StatCounter.mergeStats)
#>>> stats.minValue, stats.maxValue

#from pyspark.statcounter import StatCounter
# value[0] is the timestamp and value[1] is the float-value
# we are using two instances of StatCounter to sum-up two different statistics

#def mergeValues(s1, v1, s2, v2):
#    s1.merge(v1)
#    s2.merge(v2)
#    return

#def combineStats(s1, s2):
#    s1[0].mergeStats(s2[0])
#    s1[1].mergeStats(s2[1])
#    return
#(df.aggregateByKey((StatCounter(), StatCounter()),
#        (lambda s, values: mergeValues(s[0], values[0], s[1], values[1]),
#        (lambda s1, s2: combineStats(s1, s2))
#    .mapValues(lambda s: (  s[0].min(), s[0].max(), s[1].max(), s[1].min(), s[1].mean(), s[1].variance(), s[1].stddev,() s[1].count()))
#    .collect())