import sys
import types

from .. import NeurolangConfigParser, config


def test_switch_expression_type_printing():
    old = config.expression_type_printing()
    config.switch_expression_type_printing()
    assert (
        config.expression_type_printing()
        is not old
    )

    config.switch_expression_type_printing()
    assert (
        config.expression_type_printing()
        is old
    )

    config.enable_expression_type_printing()
    assert config.expression_type_printing()

    config.disable_expression_type_printing()
    assert not config.expression_type_printing()


def test_switch_backend_reloads_the_registered_modules(monkeypatch):
    module = types.ModuleType("neurolang_test_backend_dependent")
    module.__spec__ = None
    reloaded = []
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(
        sys.modules["neurolang.config"], "reload",
        lambda m: reloaded.append(m.__name__)
    )
    parser = NeurolangConfigParser()
    parser.reload_on_backend_switch(module.__name__)
    parser.reload_on_backend_switch(module.__name__)
    parser.reload_on_backend_switch("neurolang_test_not_imported")
    parser.switch_backend()
    assert reloaded == [module.__name__]
