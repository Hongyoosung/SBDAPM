from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class BinarySpace(_message.Message):
    __slots__ = ["shape"]
    SHAPE_FIELD_NUMBER: _ClassVar[int]
    shape: int
    def __init__(self, shape: _Optional[int] = ...) -> None: ...

class BoxSpace(_message.Message):
    __slots__ = ["dimensions", "shape_dimensions"]
    class BoxSpaceDimension(_message.Message):
        __slots__ = ["high", "low"]
        HIGH_FIELD_NUMBER: _ClassVar[int]
        LOW_FIELD_NUMBER: _ClassVar[int]
        high: float
        low: float
        def __init__(self, low: _Optional[float] = ..., high: _Optional[float] = ...) -> None: ...
    DIMENSIONS_FIELD_NUMBER: _ClassVar[int]
    SHAPE_DIMENSIONS_FIELD_NUMBER: _ClassVar[int]
    dimensions: _containers.RepeatedCompositeFieldContainer[BoxSpace.BoxSpaceDimension]
    shape_dimensions: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, dimensions: _Optional[_Iterable[_Union[BoxSpace.BoxSpaceDimension, _Mapping]]] = ..., shape_dimensions: _Optional[_Iterable[int]] = ...) -> None: ...

class DictSpace(_message.Message):
    __slots__ = ["labels", "values"]
    LABELS_FIELD_NUMBER: _ClassVar[int]
    VALUES_FIELD_NUMBER: _ClassVar[int]
    labels: _containers.RepeatedScalarFieldContainer[str]
    values: _containers.RepeatedCompositeFieldContainer[FundamentalSpace]
    def __init__(self, values: _Optional[_Iterable[_Union[FundamentalSpace, _Mapping]]] = ..., labels: _Optional[_Iterable[str]] = ...) -> None: ...

class DiscreteSpace(_message.Message):
    __slots__ = ["high"]
    HIGH_FIELD_NUMBER: _ClassVar[int]
    high: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, high: _Optional[_Iterable[int]] = ...) -> None: ...

class FundamentalSpace(_message.Message):
    __slots__ = ["binary_space", "box_space", "discrete_space"]
    BINARY_SPACE_FIELD_NUMBER: _ClassVar[int]
    BOX_SPACE_FIELD_NUMBER: _ClassVar[int]
    DISCRETE_SPACE_FIELD_NUMBER: _ClassVar[int]
    binary_space: BinarySpace
    box_space: BoxSpace
    discrete_space: DiscreteSpace
    def __init__(self, box_space: _Optional[_Union[BoxSpace, _Mapping]] = ..., discrete_space: _Optional[_Union[DiscreteSpace, _Mapping]] = ..., binary_space: _Optional[_Union[BinarySpace, _Mapping]] = ...) -> None: ...
