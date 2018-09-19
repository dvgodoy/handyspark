from pyspark.sql.types import AtomicType, ArrayType, MapType

@classmethod
def ret(cls, expr):
    return expr, cls.typeName()

AtomicType.ret = ret

def ret(self, expr):
    return expr, self.simpleString()

ArrayType.ret = ret
MapType.ret = ret
