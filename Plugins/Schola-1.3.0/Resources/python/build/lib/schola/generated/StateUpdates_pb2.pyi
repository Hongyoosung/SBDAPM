import schola.generated.Points_pb2 as _Points_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class AgentStateUpdate(_message.Message):
    __slots__ = ["actions"]
    ACTIONS_FIELD_NUMBER: _ClassVar[int]
    actions: _Points_pb2.DictPoint
    def __init__(self, actions: _Optional[_Union[_Points_pb2.DictPoint, _Mapping]] = ...) -> None: ...

class EnvironmentStep(_message.Message):
    __slots__ = ["updates"]
    class UpdatesEntry(_message.Message):
        __slots__ = ["key", "value"]
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: int
        value: AgentStateUpdate
        def __init__(self, key: _Optional[int] = ..., value: _Optional[_Union[AgentStateUpdate, _Mapping]] = ...) -> None: ...
    UPDATES_FIELD_NUMBER: _ClassVar[int]
    updates: _containers.MessageMap[int, AgentStateUpdate]
    def __init__(self, updates: _Optional[_Mapping[int, AgentStateUpdate]] = ...) -> None: ...
