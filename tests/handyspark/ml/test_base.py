import numpy as np
import numpy.testing as npt
import handyspark
from operator import itemgetter
from sklearn.preprocessing import Imputer

def test_imputer(sdf, pdf):
    hdf = sdf.toHandy
    hdf_filled = hdf.stratify(['Pclass']).fill(continuous=['Age'])
    himputer = hdf_filled.transformers.imputer()

    sdf_filled = himputer.transform(sdf)
    sage = sdf_filled.sort('PassengerId').toHandy.col['Age', None].values

    pdf_filled = []
    for pclass in [1, 2, 3]:
        filtered = pdf.query('Pclass == {}'.format(pclass))[['PassengerId', 'Age']]
        imputer = Imputer(strategy='mean').fit(filtered)
        pdf_filled.append(imputer.transform(filtered))
    pdf_filled = sorted(np.concatenate(pdf_filled, axis=0), key=itemgetter(0))
    age = list(map(itemgetter(1), pdf_filled))

    npt.assert_array_equal(sage, age)

def test_fencer(sdf, pdf):
    hdf = sdf.toHandy
    hdf_fenced = hdf.stratify(['Pclass']).fence('Fare')
    hfencer = hdf_fenced.transformers.fencer()

    sdf_fenced = hfencer.transform(sdf)
    sfare = sdf_fenced.sort('PassengerId').toHandy.col['Fare', None].values
    fences = hfencer.fences

    pdf_fenced = []
    for pclass in [1, 2, 3]:
        filtered = pdf.query('Pclass == {}'.format(pclass))[['PassengerId', 'Fare']]
        lower, upper = fences['Pclass == "{}"'.format(pclass)]['Fare']
        filtered['Fare'] = filtered['Fare'].clip(lower=lower, upper=upper)
        pdf_fenced.append(filtered)
    pdf_fenced = sorted(np.concatenate(pdf_fenced, axis=0), key=itemgetter(0))
    fare = list(map(itemgetter(1), pdf_fenced))

    npt.assert_array_equal(sfare, fare)
