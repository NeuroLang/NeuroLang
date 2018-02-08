from ..free_variable_evaluation import (
    FreeVariable, FreeVariableApplication,
    evaluate
)
import operator as op


def test_free_variable_application():
    a = FreeVariable('a')
    b = FreeVariable('b')
    assert FreeVariableApplication(op.add)(2, 3) == 5

    fva = FreeVariableApplication(op.add)(a, 3)
    fvb = FreeVariableApplication(op.sub)(fva, 10)
    fvc = FreeVariableApplication(op.mul)(fvb, b)
    fvd = FreeVariableApplication(a, kwargs={'c': b})
    fve = FreeVariableApplication(a, kwargs={'c': op.add(b, 2)})

    assert a in fva._free_variables and (len(fva._free_variables) == 1)
    assert evaluate(fva, a=2) == 5
    assert a in fvb._free_variables and (len(fvb._free_variables) == 1)
    assert evaluate(fvb, a=2) == -5
    assert b in fvc._free_variables and (len(fvc._free_variables) == 2)
    assert isinstance(evaluate(fvc, b=2), FreeVariableApplication)
    assert evaluate(evaluate(fvc, b=3), a=2) == evaluate(fvc, a=2, b=3) == -15
    assert evaluate(
        evaluate(fvd, a=lambda *args, **kwargs: kwargs['c']),
        b=2
    ) == 2
    assert evaluate(
        evaluate(fvd, b=2),
        a=lambda *args, **kwargs: kwargs['c']
    ) == 2
    assert evaluate(
        evaluate(fve, b=2),
        a=lambda *args, **kwargs: kwargs['c']
    ) == 4
    assert FreeVariableApplication(op.add)(2, 3) == 5


def test_free_variable_method_and_operator():
    a = FreeVariable('a')
    fva = a.__len__()
    fvb = a - 4
    fvc = 4 - a
    fve = a[2]

    assert evaluate(fva, a={1}) == 1
    assert evaluate(fvb, a=1) == -3
    assert evaluate(fvc, a=1) == 3
    assert evaluate(fvc * 2, a=1) == 6
    assert evaluate(a | False, a=True)
    assert evaluate(fve, a=[0, 1, 2]) == 2
