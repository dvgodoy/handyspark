from operator import itemgetter
from pyspark.mllib.evaluation import BinaryClassificationMetrics, MulticlassMetrics
from pyspark.sql import SQLContext

def thresholds(self):
    """
    * Returns thresholds in descending order.
    """
    return self.call('thresholds')

def roc(self):
    """Calls the `roc` method from the Java class

    * Returns the receiver operating characteristic (ROC) curve,
    * which is an RDD of (false positive rate, true positive rate)
    * with (0.0, 0.0) prepended and (1.0, 1.0) appended to it.
    * @see <a href="http://en.wikipedia.org/wiki/Receiver_operating_characteristic">
    * Receiver operating characteristic (Wikipedia)</a>
    """
    return self.call2('roc')

def pr(self):
    """Calls the `pr` method from the Java class

    * Returns the precision-recall curve, which is an RDD of (recall, precision),
    * NOT (precision, recall), with (0.0, p) prepended to it, where p is the precision
    * associated with the lowest recall on the curve.
    * @see <a href="http://en.wikipedia.org/wiki/Precision_and_recall">
    * Precision and recall (Wikipedia)</a>
    """
    return self.call2('pr')

def fMeasureByThreshold(self, beta=1.0):
    """Calls the `fMeasureByThreshold` method from the Java class

    * Returns the (threshold, F-Measure) curve.
    * @param beta the beta factor in F-Measure computation.
    * @return an RDD of (threshold, F-Measure) pairs.
    * @see <a href="http://en.wikipedia.org/wiki/F1_score">F1 score (Wikipedia)</a>
    """
    return self.call2('fMeasureByThreshold', beta)

def precisionByThreshold(self):
    """Calls the `precisionByThreshold` method from the Java class

    * Returns the (threshold, precision) curve.
    """
    return self.call2('precisionByThreshold')

def recallByThreshold(self):
    """Calls the `recallByThreshold` method from the Java class

    * Returns the (threshold, recall) curve.
    """
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
    """Returns confusion matrix: predicted classes are in columns,
    they are ordered by class label ascending, as in "labels".

    Predicted classes are computed according to informed threshold.

    Parameters
    ----------
    threshold: double, optional
        Threshold probability for the positive class.
        Default is 0.5.

    Returns
    -------
    confusionMatrix: DenseMatrix
    """
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