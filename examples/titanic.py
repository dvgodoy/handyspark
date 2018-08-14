import findspark
findspark.init()

from handyspark.sql import HandyFrame
from pyspark.sql import SparkSession
import matplotlib.pyplot as plt

spark = SparkSession.builder.getOrCreate()

sdf = spark.read.csv('../rawdata/train.csv', header=True, inferSchema=True)
hdf = HandyFrame(sdf)
print(hdf.str.find('Name', sub='Mr.', alias='FindMr').take(1))
hdf = hdf.fill('Age', strategy='median').fillna({'Embarked': 'S'})
print(hdf.handy._imputed_values)
print(hdf.corr_matrix())

fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(12, 3))

hdf.hist('Embarked', ax=ax1)
hdf.hist('Fare', ax=ax2)
hdf.boxplot('Age', ax=ax4)

fig.tight_layout()
fig.savefig('eda.png', format='png')
