from pyspark.mllib.common import _java2py, _py2java, JavaModelWrapper

def call2(self, name, *a):
    serde = self._sc._jvm.org.apache.spark.mllib.api.python.SerDe
    args = [_py2java(self._sc, a) for a in a]
    java_res = getattr(self._java_model, name)(*args)
    java_res = serde.fromTuple2RDD(java_res)
    res = _java2py(self._sc, java_res)
    return res

JavaModelWrapper.call2 = call2
