import findspark
findspark.init()

from handyspark.sql import HandyFrame, Bucket
from handyspark.extensions import *
from pyspark.sql import SparkSession
import matplotlib.pyplot as plt

from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.pipeline import Pipeline
from pyspark.ml.feature import VectorAssembler

spark = SparkSession.builder.getOrCreate()
sdf = spark.read.csv('../rawdata/train.csv', header=True, inferSchema=True)

assem = VectorAssembler(inputCols=['Pclass', 'Age', 'Fare'], outputCol='features')
rf = RandomForestClassifier(labelCol='Survived')
pipe = Pipeline(stages=[assem, rf])
data = sdf.select('Pclass', 'Age', 'Fare', 'Survived').dropna()
model = pipe.fit(data)
pred = model.transform(data)

bcm = BinaryClassificationMetrics(pred.select('probability', 'Survived').rdd.map(lambda row: (float(row.probability[1]), float(row.Survived))))
print(bcm.areaUnderROC)
df = bcm.getMetricsByThreshold()
print(df.toPandas())

#from operator import attrgetter
#pred.rdd.map(itemgetter(6)).map(attrgetter('values')).map(list).take(1)

sdf2 = spark.read.csv('../rawdata/train.csv', header=True, inferSchema=True)
hdf = HandyFrame(sdf)
hdf3 = hdf.stratify(['Pclass', 'Sex']).fill('Age', strategy='median')
hdf3.stratify([Bucket('Age', 3), 'Sex']).boxplot('Fare', figsize=(12, 6)).savefig('boxplot1.png', type='png')
hdf3.stratify(['Pclass', 'Sex']).boxplot(['Fare', 'Age'], figsize=(12, 6)).savefig('boxplot2.png', type='png')
hdf3.stratify(['Pclass', 'Sex']).hist('Fare', figsize=(12, 6)).savefig('hist1.png', type='png')
hdf3.stratify(['Pclass', 'Sex']).hist('Embarked', figsize=(12, 6)).savefig('hist2.png', type='png')
hdf3.stratify(['Pclass', 'Sex']).scatterplot('Fare', 'Age', figsize=(12, 6)).savefig('scatter.png', type='png')
print(hdf3.stratify([Bucket('Age', 5), 'Sex']).mode('Fare'))
print(hdf3.stratify(['Pclass', 'Sex']).sample(withReplacement=False, fraction=.1).show())
print(hdf3.stratify(['Pclass', 'Sex']).corr_matrix(['Fare', 'Age']))
print(hdf.handy.value_counts('Embarked'))
print(hdf.str.find('Name', sub='Mr.', alias='FindMr').take(1))
print(hdf3.handy._imputed_values)
hdf2 = hdf3.fill(sdf2)
print(hdf.corr_matrix())

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 10))
hdf = hdf.fill('Age', categorical=['Embarked'], strategy='median')#.fillna({'Embarked': 'S'})
hdf.hist('Embarked', ax=ax1)
hdf.hist('Fare', ax=ax2)
hdf.scatterplot('Fare', 'Age', ax=ax3)
hdf.boxplot('Fare', ax=ax4)

fig.tight_layout()
fig.savefig('eda.png', format='png')
