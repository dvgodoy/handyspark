from pyspark.sql.types import AtomicType, ArrayType, MapType

@classmethod
def ret(cls, expr):
    """Assigns a return type to the expression when used inside an `assign` method.
    """
    return expr, cls.typeName()

AtomicType.ret = ret

def ret(self, expr):
    """Assigns a return type to the expression when used inside an `assign` method.
    """
    return expr, self.simpleString()

ArrayType.ret = ret
MapType.ret = ret
