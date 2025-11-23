import schola.generated.Spaces_pb2 as _Spaces_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class AgentDefinition(_message.Message):
    __slots__ = ["action_space", "name", "normalize_actions", "normalize_obs", "obs_space"]
    ACTION_SPACE_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    NORMALIZE_ACTIONS_FIELD_NUMBER: _ClassVar[int]
    NORMALIZE_OBS_FIELD_NUMBER: _ClassVar[int]
    OBS_SPACE_FIELD_NUMBER: _ClassVar[int]
    action_space: _Spaces_pb2.DictSpace
    name: str
    normalize_actions: bool
    normalize_obs: bool
    obs_space: _Spaces_pb2.DictSpace
    def __init__(self, name: _Optional[str] = ..., obs_space: _Optional[_Union[_Spaces_pb2.DictSpace, _Mapping]] = ..., action_space: _Optional[_Union[_Spaces_pb2.DictSpace, _Mapping]] = ..., normalize_obs: bool = ..., normalize_actions: bool = ...) -> None: ...

class EnvironmentDefinition(_message.Message):
    __slots__ = ["agent_definitions"]
    class AgentDefinitionsEntry(_message.Message):
        __slots__ = ["key", "value"]
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: int
        value: AgentDefinition
        def __init__(self, key: _Optional[int] = ..., value: _Optional[_Union[AgentDefinition, _Mapping]] = ...) -> None: ...
    AGENT_DEFINITIONS_FIELD_NUMBER: _ClassVar[int]
    agent_definitions: _containers.MessageMap[int, AgentDefinition]
    def __init__(self, agent_definitions: _Optional[_Mapping[int, AgentDefinition]] = ...) -> None: ...

class TrainingDefinition(_message.Message):
    __slots__ = ["environment_definitions"]
    ENVIRONMENT_DEFINITIONS_FIELD_NUMBER: _ClassVar[int]
    environment_definitions: _containers.RepeatedCompositeFieldContainer[EnvironmentDefinition]
    def __init__(self, environment_definitions: _Optional[_Iterable[_Union[EnvironmentDefinition, _Mapping]]] = ...) -> None: ...
