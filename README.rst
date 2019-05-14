

.. image:: https://travis-ci.org/dvgodoy/handyspark.svg?branch=master
   :target: https://travis-ci.org/dvgodoy/handyspark
   :alt: Build Status


HandySpark
==========

Bringing pandas-like capabilities to Spark dataframes!
------------------------------------------------------

*HandySpark* is a package designed to improve *PySpark* user experience, especially when it comes to *exploratory data analysis* , including *visualization* capabilities!

It makes fetching data or computing statistics for columns really easy, returning *pandas objects* straight away.

It also leverages on the recently released *pandas UDFs* in Spark to allow for an out-of-the-box usage of common *pandas functions* in a Spark dataframe.

Moreover, it introduces the *stratify* operation, so users can perform more sophisticated analysis, imputation and outlier detection on stratified data without incurring in very computationally expensive *groupby* operations.

Finally, it brings the long missing capability of *plotting* data while retaining the advantage of performing distributed computation (unlike many tutorials on the internet, which just convert the whole dataset to pandas and then plot it - don't ever do that!).

Google Colab
------------

Eager to try it out right away? Don't wait any longer!

Open the notebook directly on Google Colab and try it yourself:


* `Exploring Titanic <https://colab.research.google.com/github/dvgodoy/handyspark/blob/master/notebooks/Exploring_Titanic.ipynb>`_

Installation
------------

To install *HandySpark* from `PyPI <https://pypi.org/project/handyspark/>`_, just type:

.. code-block:: python

   pip install handyspark

Documentation
-------------

You can find the full documentation `here <http://dvgodoy.github.com/handyspark>`_.

Quick Start
-----------

To use *HandySpark* , all you need to do is import the package and, after loading your data into a Spark dataframe, call the *toHandy()* method to get your own *HandyFrame* :

.. code-block:: python

   from pyspark.sql import SparkSession
   spark = SparkSession.builder.getOrCreate()

   from handyspark import *
   sdf = spark.read.csv('./tests/rawdata/train.csv', header=True, inferSchema=True)
   hdf = sdf.toHandy()

Fetching and plotting data
^^^^^^^^^^^^^^^^^^^^^^^^^^

Now you can easily fetch data as if you were using pandas, just use the *cols* object from your *HandyFrame* :

.. code-block:: python

   hdf.cols['Name'][:5]

It should return a pandas Series object:

.. code-block::

   0                              Braund, Mr. Owen Harris
   1    Cumings, Mrs. John Bradley (Florence Briggs Th...
   2                               Heikkinen, Miss. Laina
   3         Futrelle, Mrs. Jacques Heath (Lily May Peel)
   4                             Allen, Mr. William Henry
   Name: Name, dtype: object

If you include a list of columns, it will return a pandas DataFrame.

Due to the distributed nature of data in Spark, it is only possible to fetch the top rows of any given *HandyFrame*.

Using *cols* you have access to several pandas-like column and DataFrame based methods implemented in Spark:


* min / max / median / q1 / q3 / stddev / mode
* nunique
* value_counts
* corr
* hist
* boxplot
* scatterplot

For instance:

.. code-block:: python

   hdf.cols['Embarked'].value_counts(dropna=False)

.. code-block::

   S      644
   C      168
   Q       77
   NaN      2
   Name: Embarked, dtype: int64

You can also make some plots:

.. code-block:: python

   from matplotlib import pyplot as plt
   fig, axs = plt.subplots(1, 4, figsize=(12, 4))
   hdf.cols['Embarked'].hist(ax=axs[0])
   hdf.cols['Age'].boxplot(ax=axs[1])
   hdf.cols['Fare'].boxplot(ax=axs[2])
   hdf.cols[['Fare', 'Age']].scatterplot(ax=axs[3])


.. image:: /images/cols_plot.png
   :target: /images/cols_plot.png
   :alt: cols plots


Handy, right (pun intended!)? But things can get *even more* interesting if you use *stratify* !

Stratify
^^^^^^^^

Stratifying a HandyFrame means using a *split-apply-combine* approach. It will first split your HandyFrame according to the specified (discrete) columns, then it will apply some function to each stratum of data and finally combine the results back together.

This is better illustrated with an example - let's try the stratified version of our previous ``value_counts``\ :

.. code-block:: python

   hdf.stratify(['Pclass']).cols['Embarked'].value_counts()

.. code-block::

   Pclass  Embarked
   1       C            85
           Q             2
           S           127
   2       C            17
           Q             3
           S           164
   3       C            66
           Q            72
           S           353
   Name: value_counts, dtype: int64

Cool, isn't it? Besides, under the hood, not a single *group by* operation was performed - everything is handled using filter clauses! So, *no data shuffling* !

What if you want to *stratify* on a column containing continuous values? No problem!

.. code-block:: python

   hdf.stratify(['Sex', Bucket('Age', 2)]).cols['Embarked'].value_counts()

.. code-block::

   Sex     Age                                Embarked
   female  Age >= 0.4200 and Age < 40.2100    C            46
                                              Q            12
                                              S           154
           Age >= 40.2100 and Age <= 80.0000  C            15
                                              S            32
   male    Age >= 0.4200 and Age < 40.2100    C            53
                                              Q            11
                                              S           287
           Age >= 40.2100 and Age <= 80.0000  C            16
                                              Q             5
                                              S            81
   Name: value_counts, dtype: int64

You can use either *Bucket* or *Quantile* to discretize your data in any given number of bins!

What about *plotting* it? Yes, *HandySpark* can handle that as well!

.. code-block:: python

   hdf.stratify(['Sex', Bucket('Age', 2)]).cols['Embarked'].hist(figsize=(8, 6))


.. image:: /images/stratified_hist.png
   :target: /images/stratified_hist.png
   :alt: stratified hist


Handling missing data
^^^^^^^^^^^^^^^^^^^^^

*HandySpark* makes it very easy to spot and fill missing values. To figure if there are any missing values, just use *isnull* :

.. code-block:: python

   hdf.isnull(ratio=True)

.. code-block::

   PassengerId    0.000000
   Survived       0.000000
   Pclass         0.000000
   Name           0.000000
   Sex            0.000000
   Age            0.198653
   SibSp          0.000000
   Parch          0.000000
   Ticket         0.000000
   Fare           0.000000
   Cabin          0.771044
   Embarked       0.002245
   Name: missing(ratio), dtype: float64

Ok, now you know there are 3 columns with missing values: ``Age``\ , ``Cabin`` and ``Embarked``. It's time to fill those values up! But, let's skip ``Cabin``\ , which has 77% of its values missing!

So, ``Age`` is a continuous variable, while ``Embarked`` is a categorical variable. Let's start with the latter:

.. code-block:: python

   hdf_filled = hdf.fill(categorical=['Embarked'])

*HandyFrame* has a *fill* method which takes up to 3 arguments:


* categorical: a list of categorical variables
* continuous: a list of continuous variables
* strategy: which strategy to use for each one of the continuous variables (either ``mean`` or ``median``\ )

Categorical variables use a ``mode`` strategy by default.

But you do not need to stick with the basics anymore... you can fancy it up using *stratify* together with *fill* :

.. code-block:: python

   hdf_filled = hdf_filled.stratify(['Pclass', 'Sex']).fill(continuous=['Age'], strategy=['mean'])

How do you know which values are being used? Simple enough:

.. code-block:: python

   hdf_filled.statistics_

.. code-block::

   {'Embarked': 'S',
    'Pclass == "1" and Sex == "female"': {'Age': 34.61176470588235},
    'Pclass == "1" and Sex == "male"': {'Age': 41.28138613861386},
    'Pclass == "2" and Sex == "female"': {'Age': 28.722972972972972},
    'Pclass == "2" and Sex == "male"': {'Age': 30.74070707070707},
    'Pclass == "3" and Sex == "female"': {'Age': 21.75},
    'Pclass == "3" and Sex == "male"': {'Age': 26.507588932806325}}

There you go! The filter clauses and the corresponding imputation values!

But there is *more* - once you're with your imputation procedure, why not generate a *custom transformer* to do that for you, either on your test set or in production?

You only need to call the *imputer* method of the *transformer* object that every *HandyFrame* has:

.. code-block:: python

   imputer = hdf_filled.transformers.imputer()

In the example above, *imputer* is now a full-fledged serializable PySpark transformer! What does that mean? You can use it in your *pipeline* and *save / load* at will :-)

Detecting outliers
^^^^^^^^^^^^^^^^^^

Second only to the problem of missing data, outliers can pose a challenge for training machine learning models.

*HandyFrame* to the rescue, with its *outliers* method:

.. code-block:: python

   hdf_filled.outliers(method='tukey', k=3.)

.. code-block::

   PassengerId      0.0
   Survived         0.0
   Pclass           0.0
   Age              1.0
   SibSp           12.0
   Parch          213.0
   Fare            53.0
   dtype: float64

Currently, only `\ *Tukey's* <https://en.wikipedia.org/wiki/Outlier#Tukey's_fences>`_ method is available (I am working on Mahalanobis distance!). This method takes an optional *k* argument, which you can set to larger values (like 3) to allow for a more loose detection.

The good thing is, now we can take a peek at the data by plotting it:

.. code-block:: python

   from matplotlib import pyplot as plt
   fig, axs = plt.subplots(1, 4, figsize=(16, 4))
   hdf_filled.cols['Parch'].hist(ax=axs[0])
   hdf_filled.cols['SibSp'].hist(ax=axs[1])
   hdf_filled.cols['Age'].boxplot(ax=axs[2], k=3)
   hdf_filled.cols['Fare'].boxplot(ax=axs[3], k=3)


.. image:: /images/outliers.png
   :target: /images/outliers.png
   :alt: outliers


Let's focus on the ``Fare`` column - what can we do about it? Well, we could use Tukey's fences to, er... *fence* the outliers :-)

.. code-block:: python

   hdf_fenced = hdf_filled.fence(['Fare'])

Which values were used, you ask?

.. code-block:: python

   hdf_fenced.fences_

.. code-block::

   {'Fare': [-26.7605, 65.6563]}

It works quite similarly to the *fill* method and, I hope you guessed, it *also* gives you the ability to create the corresponding *custom transformer* :-)

.. code-block:: python

   fencer = hdf_fenced.transformers.fencer()

Pandas and more pandas!
^^^^^^^^^^^^^^^^^^^^^^^

With *HandySpark* you can feel *almost* as if you were using traditional pandas :-)

To gain access to the whole suite of available pandas functions, you need to leverage the *pandas* object of your *HandyFrame* :

.. code-block:: python

   some_ports = hdf_fenced.pandas['Embarked'].isin(values=['C', 'Q'])
   some_ports

.. code-block::

   Column<b'udf(Embarked) AS `<lambda>(Embarked,)`'>

In the example above, *HandySpark* treats the ``Embarked`` column as if it were a pandas Series and, therefore, you may call its *isin* method!

But, remember Spark has *lazy evaluation* , so the result is a *column expression* which leverages the power of *pandas UDFs* (provived that PyArrow is installed, otherwise it will fall back to traditional UDFs).

The only thing left to do is to actually *assign* the results to a new column, right?

.. code-block:: python

   hdf_fenced = hdf_fenced.assign(is_c_or_q=some_ports)
   # What's in there?
   hdf_fenced.cols['is_c_or_q'][:5]

.. code-block::

   0     True
   1    False
   2    False
   3     True
   4     True
   Name: is_c_or_q, dtype: bool

You got that right! *HandyFrame* has a very convenient *assign* method, just like in pandas!

It does not get much easier than that :-) There are several column methods available already:


* betweeen / between_time
* isin
* isna / isnull
* notna / notnull
* abs
* clip / clip_lower / clip_upper
* replace
* round / truncate
* tz_convert / tz_localize

And this is not all! Both specialized *str* and *dt* objects from pandas are available as well!

For instance, if you want to find if a given string contains another substring?

.. code-block:: python

   col_mrs = hdf_fenced.pandas['Name'].str.find(sub='Mrs.')
   hdf_fenced = hdf_fenced.assign(is_mrs=col_mrs > 0)


.. image:: /images/is_mrs.png
   :target: /images/is_mrs.png
   :alt: is mrs


There are many, many more available methods:


*String methods* :

#. contains
#. startswith / endswitch
#. match
#. isalpha / isnumeric / isalnum / isdigit / isdecimal / isspace
#. islower / isupper / istitle
#. replace
#. repeat
#. join
#. pad
#. slice / slice_replace
#. strip / lstrip / rstrip
#. wrap / center / ljust / rjust
#. translate
#. get
#. normalize
#. lower / upper / capitalize / swapcase / title
#. zfill
#. count
#. find / rfind
#. len

*Date / Datetime methods* :

#. is_leap_year / is_month_end / is_month_start / is_quarter_end / is_quarter_start / is_year_end / is_year_start
#. strftime
#. tz / time / tz_convert / tz_localize
#. day / dayofweek / dayofyear / days_in_month / daysinmonth
#. hour / microsecond / minute / nanosecond / second
#. week / weekday / weekday_name
#. month / quarter / year / weekofyear
#. date
#. ceil / floor / round
#. normalize

Your own functions
^^^^^^^^^^^^^^^^^^

The sky is the limit! You can create regular Python functions and use assign to create new columns :-)

No need to worry about turning them into *pandas UDFs* - everything is handled by *HandySpark* under the hood!

The arguments of your function (or ``lambda``\ ) should have the names of the columns you want to use. For instance, to take the ``log`` of ``Fare``\ :

.. code-block:: python

   import numpy as np
   hdf_fenced = hdf_fenced.assign(logFare=lambda Fare: np.log(Fare + 1))


.. image:: /images/logfare.png
   :target: /images/logfare.png
   :alt: logfare


You can also use multiple columns:

.. code-block:: python

   hdf_fenced = hdf_fenced.assign(fare_times_age=lambda Fare, Age: Fare * Age)

Even though the result is kinda pointless, it will work :-)

Keep in mind that the *return type* , that is, the column type of the new column, will be the same as the first column used (\ ``Fare``\ , in the example).

What if you want to return something of a *different* type?! No worries! You only need to *wrap* your function with the desired return type. An example should make this more clear:

.. code-block:: python

   from pyspark.sql.types import StringType

   hdf_fenced = hdf_fenced.assign(str_fare=StringType.ret(lambda Fare: Fare.map('${:,.2f}'.format)))

   hdf_fenced.cols['str_fare'][:5]

.. code-block::

   0    $65.66
   1    $53.10
   2    $26.55
   3    $65.66
   4    $65.66
   Name: str_fare, dtype: object

Basically, we imported the desired output type - *StringType* - and used its extended method *ret* to wrap our ``lambda`` function that formats our numeric ``Fare`` column into a string.

It is also possible to create a more complex type, like an array of doubles:

.. code-block:: python

   from pyspark.sql.types import ArrayType, DoubleType

   def make_list(Fare):
       return Fare.apply(lambda v: [v, v*2])

   hdf_fenced = hdf_fenced.assign(fare_list=ArrayType(DoubleType()).ret(make_list))

   hdf_fenced.cols['fare_list'][:5]

.. code-block::

   0           [7.25, 14.5]
   1    [71.2833, 142.5666]
   2         [7.925, 15.85]
   3          [53.1, 106.2]
   4           [8.05, 16.1]
   Name: fare_list, dtype: object

OK, so, what happened here?


#. First, we imported the necessary types, *ArrayType* and *DoubleType* , since we are building a function that returns a list of doubles.
#. We actually built the function - notice that we call *apply* straight from *Fare* , which is treated as a pandas Series under the hood.
#. We *wrap* the function with the return type ``ArrayType(DoubleType())`` by invoking the extended method ``ret``.
#. Finally, we assign it to a new column name, and that's it!

Nicer exceptions
^^^^^^^^^^^^^^^^

Now, suppose you make a mistake while creating your function... if you have used Spark for a while, you already realized that, when an exception is raised, it will be *loooong* , right?

To help you with that, *HandySpark* analyzes the error message and parses it nicely for you at the very *top* of the error message, in *bold red* :


.. image:: /images/handy_exception.png
   :target: /images/handy_exception.png
   :alt: exception


Safety first
^^^^^^^^^^^^

*HandySpark* wants to protect your cluster and network, so it implements a *safety* whenever you perform an operation that are going to retrieve *ALL* data from your *HandyFrame* , like ``collect`` or ``toPandas``.

How does that work? Every time a *HandyFrame* has one of these methods called, it will output up to the *safety limit* , which has a default of *1,000 elements*.


.. image:: /images/safety_on.png
   :target: /images/safety_on.png
   :alt: safety on


Do you want to set a different safety limit for your *HandyFrame* ?


.. image:: /images/safety_limit.png
   :target: /images/safety_limit.png
   :alt: safety limit


What if you want to retrieve everything nonetheless?! You can invoke the *safety_off* method prior to the actual method you want to call and you get a *one-time* unlimited result.


.. image:: /images/safety_off.png
   :target: /images/safety_off.png
   :alt: safety off


Don't feel like Handy anymore?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To get back your original Spark dataframe, you only need to call *notHandy* to make it not handy again:

.. code-block:: python

   hdf_fenced.notHandy()

.. code-block::

   DataFrame[PassengerId: int, Survived: int, Pclass: int, Name: string, Sex: string, Age: double, SibSp: int, Parch: int, Ticket: string, Fare: double, Cabin: string, Embarked: string, logFare: double, is_c_or_q: boolean]

Comments, questions, suggestions, bugs
--------------------------------------

*DISCLAIMER* : this is a project *under development* , so it is likely you'll run into bugs/problems.

So, if you find any bugs/problems, please open an `issue <https://github.com/dvgodoy/handyspark/issues>`_ or submit a `pull request <https://github.com/dvgodoy/handyspark/pulls>`_.
