import numpy as np
import numpy.testing as npt
import pandas as pd
import handyspark
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.pipeline import Pipeline
from pyspark.mllib.evaluation import BinaryClassificationMetrics
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve

def test_confusion_matrix(sdf):
    assem = VectorAssembler(inputCols=['Fare', 'Pclass', 'Age'], outputCol='features')
    rf = RandomForestClassifier(featuresCol='features', labelCol='Survived', numTrees=20)
    pipeline = Pipeline(stages=[assem, rf])
    model = pipeline.fit(sdf.fillna(0.0))
    predictions = model.transform(sdf.fillna(0.0)).toHandy.to_metrics_RDD('probability', 'Survived')
    bcm = BinaryClassificationMetrics(predictions)
    predictions = np.array(predictions.collect())

    scm = bcm.confusionMatrix().toArray()
    pcm = confusion_matrix(predictions[:, 1], predictions[:, 0] > .5)
    npt.assert_array_almost_equal(scm, pcm)

    scm = bcm.confusionMatrix(.3).toArray()
    pcm = confusion_matrix(predictions[:, 1], predictions[:, 0] > .3)
    npt.assert_array_almost_equal(scm, pcm)

def test_get_metrics_by_threshold(sdf):
    assem = VectorAssembler(inputCols=['Fare', 'Pclass', 'Age'], outputCol='features')
    rf = RandomForestClassifier(featuresCol='features', labelCol='Survived', numTrees=20, seed=13)
    pipeline = Pipeline(stages=[assem, rf])
    model = pipeline.fit(sdf.fillna(0.0))
    predictions = model.transform(sdf.fillna(0.0)).toHandy.to_metrics_RDD('probability', 'Survived')
    bcm = BinaryClassificationMetrics(predictions)
    metrics = bcm.getMetricsByThreshold()

    predictions = np.array(predictions.collect())

    pr = np.array(bcm.pr().collect())
    idx = pr[:, 0].argmax()
    pr = pr[:idx + 1, :]
    precision, recall, thresholds = precision_recall_curve(predictions[:, 1], predictions[:, 0])

    npt.assert_array_almost_equal(precision, pr[:, 1][::-1])
    npt.assert_array_almost_equal(recall, pr[:, 0][::-1])

    roc = np.array(bcm.roc().collect())
    idx = roc[:, 1].argmax()
    roc = roc[:idx + 1, :]
    sroc = pd.DataFrame(np.round(roc, 6), columns=['fpr', 'tpr'])
    sroc = sroc.groupby('fpr').agg({'tpr': [np.min, np.max]})

    fpr, tpr, thresholds = roc_curve(predictions[:, 1], predictions[:, 0])
    idx = tpr.argmax()
    proc = pd.DataFrame({'fpr': np.round(fpr[:idx + 1], 6), 'tpr': np.round(tpr[:idx + 1], 6)})
    proc = proc.groupby('fpr').agg({'tpr': [np.min, np.max]})

    sroc = sroc.join(proc, how='inner', rsuffix='sk')

    npt.assert_array_almost_equal(sroc.iloc[:, 0], proc.iloc[:, 0])
    npt.assert_array_almost_equal(sroc.iloc[:, 1], proc.iloc[:, 1])
