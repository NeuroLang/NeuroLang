from .... import frontend as fe


def test_add_tuple_set_length():
    nl = fe.NeurolangDL()
    test = ['test0', 'test1', 'test2']
    nl.add_tuple_set(((e,) for e in test), name='test')
    assert len(nl.symbols['test']) == 3
