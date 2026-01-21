from __future__ import annotations

from types import SimpleNamespace


class _ComponentFactory:
    def __getattr__(self, name):
        def _factory(*args, **kwargs):
            return {"type": name, "args": args, "props": kwargs}
        return _factory


class exceptions:
    class PreventUpdate(Exception):
        pass


no_update = object()
ctx = SimpleNamespace(triggered=[], triggered_id=None)
callback_context = ctx


def Input(*args, **kwargs):
    return ("Input", args, kwargs)


def Output(*args, **kwargs):
    return ("Output", args, kwargs)


def State(*args, **kwargs):
    return ("State", args, kwargs)


class _DataTable:
    def __init__(self, *args, **kwargs):
        self.props = {"columns": kwargs.get("columns", []), "data": kwargs.get("data", [])}
        self.type = "DataTable"

    def to_plotly_json(self):
        return {"type": self.type, "props": dict(self.props)}

    def __getattr__(self, name):
        if name == "data":
            return self.props.get("data", [])
        raise AttributeError(name)


def _datatable(*args, **kwargs):
    return _DataTable(*args, **kwargs)


dash_table = SimpleNamespace(DataTable=_datatable)

html = _ComponentFactory()
dcc = _ComponentFactory()
