Static vs Dynamic Environments
==============================

Schola provides two types of environments for reinforcement learning: **Static** and **Dynamic**. These environments differ in how agents and trainers are managed, with the Static Environment 

Static Environments
-------------------

Static environments are designed for scenarios where the agents and trainers are predefined and do not change dynamically during the simulation. These environments are more rigid and are suitable for use cases where the structure of the environment remains constant. To create a static environment, you can use the :cpp:class:`AStaticScholaEnvironment` class, which is a subclass of :cpp:class:`AbstractScholaEnvironment <AAbstractScholaEnvironment>`.

Key Differences
~~~~~~~~~~~~~~~
- **Agent Registration**: Agents are registered using pointers to AActor's controlled by :cpp:class:`AbstractTrainer <AAbstractTrainer>`. This method is implemented by child classes.


Dynamic Environments
--------------------

Dynamic environments are designed for scenarios where agents and trainers can be created, modified, or removed during the simulation. These environments are more flexible and are suitable for use cases where the structure of the environment evolves over time. An example would be a zombie game where a variable number of zombies are spawned throughout the game. To create a dynamic environment, you can use the :cpp:class:`DynamicScholaEnvironment <ADynamicScholaEnvironment>` class, which is a subclass of :cpp:class:`AbstractScholaEnvironment <AAbstractScholaEnvironment>`.

Key Differences
~~~~~~~~~~~~~~~
- **Agent Registration**: Agents are registered using a pair of UClass Objects, one a child of :cpp:class:`AbstractTrainer <AAbstractTrainer>` and the other a child of `Pawn (Object Reference) <https://dev.epicgames.com/documentation/en-us/unreal-engine/pawn-in-unreal-engine>`_. This method can be overridden in Blueprints.
- **Agent Spawning**: Agents can be spawned using the :cpp:func:`~ADynamicScholaEnvironment::SpawnAgent` method. This allows for more complex and adaptive environments where agents can be added or removed based on the simulation's needs.


