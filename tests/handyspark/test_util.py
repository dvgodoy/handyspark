import findspark
import os
import pandas as pd
import pytest
import numpy as np
import numpy.testing as npt
from pyspark.sql import SparkSession

FIXTURE_DIR = os.path.join(os.path.split(os.path.dirname(os.path.realpath(__file__)))[0], 'rawdata')

@pytest.fixture(scope='module')
def sdf():
    findspark.init()
    spark = SparkSession.builder.getOrCreate()

    sdf = spark.read.csv(os.path.join(FIXTURE_DIR, 'train.csv'), header=True, inferSchema=True)
    return sdf

@pytest.fixture(scope='module')
def pdf():
    pdf = pd.read_csv(os.path.join(FIXTURE_DIR, 'train.csv'), header=True)
    return pdf

