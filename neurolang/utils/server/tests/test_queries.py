import time
from concurrent.futures import Future, TimeoutError
from unittest.mock import create_autospec
from uuid import uuid4

import pytest
from lark.exceptions import UnexpectedCharacters
from neurolang.frontend.probabilistic_frontend import NeurolangPDL
from neurolang.exceptions import NeuroLangException
from ..app import NeurolangQueryManager
from ..engines import NeurolangEngineConfiguration


@pytest.fixture
def conf():
    conf = create_autospec(NeurolangEngineConfiguration)
    conf.key = "mockengine"
    conf.create.side_effect = lambda : NeurolangPDL()
    return conf

def test_nqm_creates_engines(conf):
    opts = {conf: 5}
    nqm = NeurolangQueryManager(opts)
    time.sleep(.5)

    assert conf.key in nqm.engines
    assert nqm.engines[conf.key].counter == 5
    assert len(nqm.engines[conf.key].engines) == 5
    assert conf.create.call_count == 5

def test_nqm_submits_queries(conf):
    opts = {conf: 2}
    nqm = NeurolangQueryManager(opts)
    time.sleep(.2)

    uuid = uuid4()
    query = "invalid query"
    res = nqm.submit_query(uuid, query, conf.key)
    assert isinstance(res, Future)
    assert res is nqm.get_result(uuid)

    # wait for future execution. It should raise the exception
    with pytest.raises(UnexpectedCharacters):
        res.result()
        raise NeuroLangException

def test_nqm_waits_for_engines_to_be_available(conf):
    # create two engines
    opts = {conf: 2}
    nqm = NeurolangQueryManager(opts)
    time.sleep(.2)

    # acquire the two engines manually to block execution
    nqm.engines[conf.key].sema.acquire()
    nqm.engines[conf.key].sema.acquire()

    # submit three tasks
    id1 = uuid4()
    id2 = uuid4()
    id3 = uuid4()
    res1 = nqm.submit_query(id1, "A('x', 3)", conf.key)
    res2 = nqm.submit_query(id2, "A('y', 2)", conf.key)
    res3 = nqm.submit_query(id3, "A('z', 1)", conf.key)
    assert isinstance(res1, Future)
    assert isinstance(res2, Future)
    assert isinstance(res3, Future)
    
    # wait 1sec for 1st query to complete
    with pytest.raises(TimeoutError):
        res1.result(timeout=1)
    # q1 & q2 should be running but not done. q3 should not be running.
    assert not res1.done()
    assert not res2.done()
    assert res1.running()
    assert res2.running()
    assert not res3.running()

    # release an engine
    nqm.engines[conf.key].sema.release()
    # all 3 queries should complete
    res1 = nqm.get_result(id1)
    res2 = nqm.get_result(id2)
    res3 = nqm.get_result(id3)
    assert res1.result() is not None
    assert res2.result() is not None
    assert res3.result() is not None

def test_nqm_cancels_queries(conf):
    # create one engine
    opts = {conf: 1}
    nqm = NeurolangQueryManager(opts)
    time.sleep(.1)

    # acquire the engine manually to block execution
    nqm.engines[conf.key].sema.acquire()

    # submit two tasks
    id1 = uuid4()
    id2 = uuid4()
    res1 = nqm.submit_query(id1, "A('x', 3)", conf.key)
    res2 = nqm.submit_query(id2, "A('y', 2)", conf.key)
    time.sleep(.1)
    assert not res1.done()
    assert res1.running()
    assert not res2.running()

    # cancel the second task
    cancelled = nqm.cancel(id2)
    time.sleep(.1)
    assert cancelled
    res2 = nqm.get_result(id2)
    assert not res2.running()
    assert res2.cancelled()

    # release engine to finish first task
    nqm.engines[conf.key].sema.release()
    assert res1.result() is not None
