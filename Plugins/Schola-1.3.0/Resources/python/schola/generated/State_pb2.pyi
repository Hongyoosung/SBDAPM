import schola.generated.Points_pb2 as _Points_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

COMPLETED: Status
DESCRIPTOR: _descriptor.FileDescriptor
RUNNING: Status
TRUNCATED: Status

class AgentState(_message.Message):
    __slots__ = ["info", "observations", "reward", "status"]
    class InfoEntry(_message.Message):
        __slots__ = ["key", "value"]
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    INFO_FIELD_NUMBER: _ClassVar[int]
    OBSERVATIONS_FIELD_NUMBER: _ClassVar[int]
    REWARD_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    info: _containers.ScalarMap[str, str]
    observations: _Points_pb2.DictPoint
    reward: float
    status: Status
    def __init__(self, observations: _Optional[_Union[_Points_pb2.DictPoint, _Mapping]] = ..., reward: _Optional[float] = ..., status: _Optional[_Union[Status, str]] = ..., info: _Optional[_Mapping[str, str]] = ...) -> None: ...

class EnvironmentState(_message.Message):
    __slots__ = ["agent_states"]
    class AgentStatesEntry(_message.Message):
        __slots__ = ["key", "value"]
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: int
        value: AgentState
        def __init__(self, key: _Optional[int] = ..., value: _Optional[_Union[AgentState, _Mapping]] = ...) -> None: ...
    AGENT_STATES_FIELD_NUMBER: _ClassVar[int]
    agent_states: _containers.MessageMap[int, AgentState]
    def __init__(self, agent_states: _Optional[_Mapping[int, AgentState]] = ...) -> None: ...

class InitialAgentState(_message.Message):
    __slots__ = ["info", "observations"]
    class InfoEntry(_message.Message):
        __slots__ = ["key", "value"]
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    INFO_FIELD_NUMBER: _ClassVar[int]
    OBSERVATIONS_FIELD_NUMBER: _ClassVar[int]
    info: _containers.ScalarMap[str, str]
    observations: _Points_pb2.DictPoint
    def __init__(self, observations: _Optional[_Union[_Points_pb2.DictPoint, _Mapping]] = ..., info: _Optional[_Mapping[str, str]] = ...) -> None: ...

class InitialEnvironmentState(_message.Message):
    __slots__ = ["agent_states"]
    class AgentStatesEntry(_message.Message):
        __slots__ = ["key", "value"]
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: int
        value: InitialAgentState
        def __init__(self, key: _Optional[int] = ..., value: _Optional[_Union[InitialAgentState, _Mapping]] = ...) -> None: ...
    AGENT_STATES_FIELD_NUMBER: _ClassVar[int]
    agent_states: _containers.MessageMap[int, InitialAgentState]
    def __init__(self, agent_states: _Optional[_Mapping[int, InitialAgentState]] = ...) -> None: ...

class InitialTrainingState(_message.Message):
    __slots__ = ["environment_states"]
    class EnvironmentStatesEntry(_message.Message):
        __slots__ = ["key", "value"]
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: int
        value: InitialEnvironmentState
        def __init__(self, key: _Optional[int] = ..., value: _Optional[_Union[InitialEnvironmentState, _Mapping]] = ...) -> None: ...
    ENVIRONMENT_STATES_FIELD_NUMBER: _ClassVar[int]
    environment_states: _containers.MessageMap[int, InitialEnvironmentState]
    def __init__(self, environment_states: _Optional[_Mapping[int, InitialEnvironmentState]] = ...) -> None: ...

class TrainingState(_message.Message):
    __slots__ = ["environment_states"]
    ENVIRONMENT_STATES_FIELD_NUMBER: _ClassVar[int]
    environment_states: _containers.RepeatedCompositeFieldContainer[EnvironmentState]
    def __init__(self, environment_states: _Optional[_Iterable[_Union[EnvironmentState, _Mapping]]] = ...) -> None: ...

class Status(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = []
