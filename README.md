[![Build Status](https://travis-ci.org/dvgodoy/handyspark.svg?branch=master)](https://travis-ci.org/dvgodoy/handyspark)
[![Documentation Status](http://readthedocs.org/projects/handyspark/badge/?version=latest)](http://handyspark.readthedocs.io/en/latest/?badge=latest)

# HandySpark

## Bringing pandas-like capabilities to Spark dataframes!

***HandySpark*** is a package designed to improve ***PySpark*** user experience, especially when it comes to ***exploratory data analysis***, including ***visualization*** capabilities, which are completely absent in the original API.

It makes fetching data or computing statistics for columns really easy, returning ***pandas objects*** straight away.

It also leverages on the recently release ***pandas UDFs*** in Spark to allow for a smooth usage of common ***pandas functions*** out-of-the-box in a Spark dataframe.

Moreover, it introduces the ***stratify*** operation, so users can perform more sophisticated analysis, imputation and outlier detection on stratified data without incurring in very computationally expensive ***groupby*** operations.

Finally, it brings the long missing capability of ***plotting*** data while retaining the advantage of performing distributed computation (unlike many tutorials on the internet, which just convert the whole dataset to pandas and then plot it - don't ever do that!).

### Google Colab

Eager to try it out right away? Don't wait any longer!

Open the notebook directly on Google Colab and try it yourself:

- [Exploring Titanic](https://colab.research.google.com/github/dvgodoy/handyspark/blob/master/notebooks/Exploring_Titanic.ipynb)

### Installation

To install ***HandySpark*** from [PyPI](https://pypi.org/project/handyspark/), just type:
```python
pip install handyspark
```

### Documentation

You can find the full documentations at [Read the Docs](http://handyspark.readthedocs.io/).

### Quick Start

To use ***HandySpark***, all you need to do is import the package and, after loading your data into a Spark dataframe, call the ***toHandy()*** method to get your own ***HandyFrame***:
```python
from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()

from handyspark import *
sdf = spark.read.csv('./tests/rawdata/train.csv', header=True, inferSchema=True)
hdf = sdf.toHandy()
```

Now you can easily fetch data as if you were using pandas, just use the ***cols*** object from your ***HandyFrame***:
```python
hdf.cols['Name'][:5]
```

Should return a pandas Series object:
```
0                              Braund, Mr. Owen Harris
1    Cumings, Mrs. John Bradley (Florence Briggs Th...
2                               Heikkinen, Miss. Laina
3         Futrelle, Mrs. Jacques Heath (Lily May Peel)
4                             Allen, Mr. William Henry
Name: Name, dtype: object
```

If you include a list of columns, it will return a pandas DataFrame.

Due to the distributed nature of data in Spark, it is only possible to fetch the top rows of any given ***HandyFrame***.

Using ***cols*** you have access to several pandas-like column and DataFrame based methods implemented in Spark:

- min / max / median / q1 / q3 / stddev / mode
- nunique
- value_counts
- corr
- hist
- boxplot
- scatterplot

For instance:
```python
hdf.cols['Embarked'].value_counts(dropna=False)
```

Should return:
```
S      644
C      168
Q       77
NaN      2
Name: Embarked, dtype: int64
```

You can also make some plots:
```python
fig, axs = plt.subplots(1, 3, figsize=(12, 4))
hdf.cols['Embarked'].hist(ax=axs[0])
hdf.cols['Age'].boxplot(ax=axs[1])
hdf.cols['Fare'].boxplot(ax=axs[2])
```

The results should look like this:

![cols plots](/images/cols_plot.png)

## Comments, questions, suggestions, bugs

***DISCLAIMER***: this is a project ***under development***, so it is likely you'll run into bugs/problems.

So, if you find any bugs/problems, please open an [issue](https://github.com/dvgodoy/handyspark/issues) or submit a [pull request](https://github.com/dvgodoy/handyspark/pulls).
