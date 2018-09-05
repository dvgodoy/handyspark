from pyspark.sql.types import AtomicType

@classmethod
def ret(cls, expr):
    return expr, cls.typeName()

AtomicType.ret = ret

# TO DO - UDTs, arrays, structs and map