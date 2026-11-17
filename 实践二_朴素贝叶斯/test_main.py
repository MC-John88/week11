import numpy as np
import pytest
from common import validate_data,evaluate
from main import build_model,split_data

@pytest.fixture
def data():
    rng=np.random.default_rng(42)
    return rng.normal(size=(40,2)),np.array([0,1]*20)

def test_split_reproducible(data):
    a=split_data(*data);b=split_data(*data)
    assert all(np.array_equal(x,y) for x,y in zip(a,b))

def test_train_predict_probability(data):
    xtr,xte,ytr,yte=split_data(*data);m=build_model().fit(xtr,ytr)
    assert len(m.predict(xte))==len(yte)
    p=m.predict_proba(xte)
    assert np.all((p>=0)&(p<=1)) and np.allclose(p.sum(axis=1),1)

def test_evaluation():
    r=evaluate([0,0,1,1],[0,1,1,1]);assert r['accuracy']==.75
    assert r['confusion_matrix']==[[1,1],[0,2]]

@pytest.mark.parametrize('x,y',[(np.empty((0,2)),np.array([])),(np.ones((3,2)),np.array([0,1])),(np.ones((3,2)),np.array([0,1,2])),(np.array([[float('nan')],[1.]]),np.array([0,1]))])
def test_invalid_data(x,y):
    with pytest.raises(ValueError):validate_data(x,y)

def test_prediction_dimension_error(data):
    m=build_model().fit(*data)
    with pytest.raises(ValueError):m.predict(np.ones((2,3)))

def test_invalid_smoothing():
    with pytest.raises(ValueError):build_model(var_smoothing=0)
