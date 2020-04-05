from sqlalchemy.schema import DDLElement
from sqlalchemy.sql import table
from sqlalchemy.ext import compiler


class CreateView(DDLElement):
    def __init__(self, name, selectable):
        self.name = name
        self.selectable = selectable


class DropView(DDLElement):
    def __init__(self, name):
        self.name = name


@compiler.compiles(CreateView)
def compile_cv(element, compiler, **kw):
    return "CREATE VIEW %s AS %s" % (
        element.name,
        compiler.sql_compiler.process(element.selectable, literal_binds=True))


@compiler.compiles(DropView)
def compile_dv(element, compiler, **kw):
    return "DROP VIEW %s" % (element.name)


def view(name, metadata, selectable):
    t = table(name)
    for c in selectable.c:
        c._make_proxy(t)
    CreateView(name, selectable).execute_at('after-create', metadata)
    DropView(name).execute_at('before-drop', metadata)
    return t
