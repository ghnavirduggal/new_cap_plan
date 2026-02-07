from __future__ import annotations

from types import SimpleNamespace


class _ComponentFactory:
    def __getattr__(self, name):
        def _factory(*args, **kwargs):
            return {"type": name, "args": args, "props": kwargs}
        return _factory


_component = _ComponentFactory()

Card = _component.Card
CardBody = _component.CardBody
Row = _component.Row
Col = _component.Col
Button = _component.Button
Modal = _component.Modal
ModalHeader = _component.ModalHeader
ModalTitle = _component.ModalTitle
ModalBody = _component.ModalBody
ModalFooter = _component.ModalFooter
Tabs = _component.Tabs
Tab = _component.Tab
Alert = _component.Alert
Checklist = _component.Checklist
Switch = _component.Switch
Label = _component.Label
InputGroup = _component.InputGroup
InputGroupText = _component.InputGroupText
Input = _component.Input
