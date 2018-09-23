import findspark
import os
import pandas as pd
import pytest
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier

FIXTURE_DIR = os.path.join(os.path.split(os.path.dirname(os.path.realpath(__file__)))[0], 'rawdata')

findspark.init()
spark = SparkSession.builder.getOrCreate()
df = spark.read.csv(os.path.join(FIXTURE_DIR, 'train.csv'), header=True, inferSchema=True)

@pytest.fixture(scope='module')
def sdf():
    return df

@pytest.fixture(scope='module')
def pdf():
    pdf = pd.read_csv(os.path.join(FIXTURE_DIR, 'train.csv'))
    return pdf

@pytest.fixture(scope='module')
def predicted():
    assem = VectorAssembler(inputCols=['Fare', 'Pclass', 'Age'], outputCol='features')
    feat_df = assem.transform(df.select('Fare', 'Pclass', 'Age', 'Survived').dropna())
    rf = RandomForestClassifier(featuresCol='features', labelCol='Survived')
    model = rf.fit(feat_df)
    return model.transform(feat_df)
