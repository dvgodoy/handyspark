import findspark
findspark.init()

from handyspark.sql import HandyFrame
from pyspark.sql import SparkSession
import matplotlib.pyplot as plt

spark = SparkSession.builder.getOrCreate()

sdf = spark.read.csv('../rawdata/train.csv', header=True, inferSchema=True)
sdf2 = spark.read.csv('../rawdata/train.csv', header=True, inferSchema=True)
hdf = HandyFrame(sdf)
print(hdf.handy.value_counts('Embarked'))
print(hdf.str.find('Name', sub='Mr.', alias='FindMr').take(1))
hdf = hdf.fill('Age', categorical=['Embarked'], strategy='median', strata=['Pclass', 'Sex'])#.fillna({'Embarked': 'S'})
print(hdf.handy._imputed_values)
hdf2 = hdf.fill(sdf2)
print(hdf.corr_matrix())

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 10))

hdf.hist('Embarked', ax=ax1)
hdf.hist('Fare', ax=ax2)
hdf.scatterplot('Fare', 'Age', ax=ax3)
hdf.boxplot('Age', ax=ax4)

fig.tight_layout()
fig.savefig('eda.png', format='png')
