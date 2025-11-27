from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class BinaryPoint(_message.Message):
    __slots__ = ["values"]
    VALUES_FIELD_NUMBER: _ClassVar[int]
    values: _containers.RepeatedScalarFieldContainer[bool]
    def __init__(self, values: _Optional[_Iterable[bool]] = ...) -> None: ...

class BoxPoint(_message.Message):
    __slots__ = ["values"]
    VALUES_FIELD_NUMBER: _ClassVar[int]
    values: _containers.RepeatedScalarFieldContainer[float]
    def __init__(self, values: _Optional[_Iterable[float]] = ...) -> None: ...

class DictPoint(_message.Message):
    __slots__ = ["values"]
    VALUES_FIELD_NUMBER: _ClassVar[int]
    values: _containers.RepeatedCompositeFieldContainer[FundamentalPoint]
    def __init__(self, values: _Optional[_Iterable[_Union[FundamentalPoint, _Mapping]]] = ...) -> None: ...

class DiscretePoint(_message.Message):
    __slots__ = ["values"]
    VALUES_FIELD_NUMBER: _ClassVar[int]
    values: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, values: _Optional[_Iterable[int]] = ...) -> None: ...

class FundamentalPoint(_message.Message):
    __slots__ = ["binary_point", "box_point", "discrete_point"]
    BINARY_POINT_FIELD_NUMBER: _ClassVar[int]
    BOX_POINT_FIELD_NUMBER: _ClassVar[int]
    DISCRETE_POINT_FIELD_NUMBER: _ClassVar[int]
    binary_point: BinaryPoint
    box_point: BoxPoint
    discrete_point: DiscretePoint
    def __init__(self, box_point: _Optional[_Union[BoxPoint, _Mapping]] = ..., discrete_point: _Optional[_Union[DiscretePoint, _Mapping]] = ..., binary_point: _Optional[_Union[BinaryPoint, _Mapping]] = ...) -> None: ...
