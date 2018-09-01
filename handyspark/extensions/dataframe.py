from pyspark.sql import DataFrame
from handyspark.sql import HandyFrame

@property
def handy(self):
    return HandyFrame(self)

DataFrame.handy = handy