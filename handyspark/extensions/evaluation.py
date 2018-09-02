from operator import itemgetter
from pyspark.mllib.evaluation import BinaryClassificationMetrics, MulticlassMetrics
from pyspark.sql import SQLContext

def thresholds(self):
    return self.call('thresholds')

def roc(self):
    return self.call2('roc')

def pr(self):
    return self.call2('pr')

def fMeasureByThreshold(self, beta=1.0):
    return self.call2('fMeasureByThreshold', beta)

def precisionByThreshold(self):
    return self.call2('precisionByThreshold')

def recallByThreshold(self):
    return self.call2('recallByThreshold')

def getMetricsByThreshold(self):
    thresholds = self.call('thresholds').collect()
    roc = self.call2('roc').collect()[1:-1]
    pr = self.call2('pr').collect()[1:]
    metrics = list(zip(thresholds, map(itemgetter(0), roc), map(itemgetter(1), roc), map(itemgetter(1), pr)))
    metrics += [(0., 1., 1., 0.)]
    sql_ctx = SQLContext.getOrCreate(self._sc)
    df = sql_ctx.createDataFrame(metrics).toDF('threshold', 'fpr', 'recall', 'precision')
    return df

def confusionMatrix(self, threshold=0.5):
    scoreAndLabels = self.call2('scoreAndLabels').map(lambda t: (float(t[0] > threshold), t[1]))
    mcm = MulticlassMetrics(scoreAndLabels)
    return mcm.confusionMatrix()

BinaryClassificationMetrics.thresholds = thresholds
BinaryClassificationMetrics.roc = roc
BinaryClassificationMetrics.pr = pr
BinaryClassificationMetrics.fMeasureByThreshold = fMeasureByThreshold
BinaryClassificationMetrics.precisionByThreshold = precisionByThreshold
BinaryClassificationMetrics.recallByThreshold = recallByThreshold
BinaryClassificationMetrics.getMetricsByThreshold = getMetricsByThreshold
BinaryClassificationMetrics.confusionMatrix = confusionMatrix