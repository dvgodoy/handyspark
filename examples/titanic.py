import findspark
findspark.init()

from handyspark.sql import HandyFrame, Bucket
from pyspark.sql import SparkSession
import matplotlib.pyplot as plt

spark = SparkSession.builder.getOrCreate()
sdf = spark.read.csv('../rawdata/train.csv', header=True, inferSchema=True)
sdf2 = spark.read.csv('../rawdata/train.csv', header=True, inferSchema=True)
hdf = HandyFrame(sdf)
hdf3 = hdf.stratify([Bucket('Fare'), 'Sex']).fill('Age', strategy='median')
print(hdf3.stratify(['Pclass', 'Sex']).mode('Fare'))
print(hdf3.stratify(['Pclass', 'Sex']).sample(withReplacement=False, fraction=.1).show())
print(hdf3.stratify(['Pclass', 'Sex']).corr_matrix(['Fare', 'Age']))
print(hdf.handy.value_counts('Embarked'))
print(hdf.str.find('Name', sub='Mr.', alias='FindMr').take(1))
# hdf = hdf.fill('Age', categorical=['Embarked'], strategy='median')#.fillna({'Embarked': 'S'})
print(hdf3.handy._imputed_values)
hdf2 = hdf3.fill(sdf2)
print(hdf.corr_matrix())

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 10))

hdf.hist('Embarked', ax=ax1)
hdf.hist('Fare', ax=ax2)
hdf.scatterplot('Fare', 'Age', ax=ax3)
hdf.boxplot('Age', ax=ax4)

fig.tight_layout()
fig.savefig('eda.png', format='png')
