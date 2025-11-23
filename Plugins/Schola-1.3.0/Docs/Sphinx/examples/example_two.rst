Building Ball Shooter
=====================

This guide walks you through creating a simple shooting range environment and training an agent to shoot moving targets in Unreal Engine. The goal is to aim correctly before shooting and take down targets one by one.


.. image:: /_static/guides/example_two/BallShooter.gif

In this example, we will create a dynamic shooting range environment with a single agent that learns to shoot moving targets using reinforcement learning. The agent interacts with the environment by observing the target's movements and performing actions through actuators to rotate and shoot.

We will train our agent by having it repeatedly attempt to hit the moving targets. Each attempt is referred to as an episode and ends when the agent successfully hits the target three times, or runs out of time.

Periodically, the agent will review its performance during previous episodes and update its policy to improve further. To quantify the agent's performance, we define a reward function: hitting the target earns a reward, missing incurs a penalty, and each step taken without hitting the target incurs a small penalty. The agent can then use the learned policy to decide which actions to take during gameplay.


The Structure of the Environment in Unreal Engine
-------------------------------------------------

To build the game (called environment hereafter) where the agent will learn to shoot moving targets, we need the following in our Unreal Engine project:  

-  :localref:`Map<Creating the Map>`: The game map includes the floor, four walls, the agent, and the environment definition.  
-  :localref:`Ball Blueprint<Creating the Ball>`: The projectile that the agent shoots, which is spawned by the agent when it takes the shoot action.
-  :localref:`Target Blueprint<Creating the Target>`: The object that the agent will shoot, which moves randomly around the map and is destroyed when hit three times by a ball.
-  :localref:`Agent blueprint<Creating the Agent>`: A subclass of `Pawn <https://dev.epicgames.com/documentation/en-us/unreal-engine/pawn-in-unreal-engine>`_, which includes the shape and appearance of the agent.  
-  :localref:`Shooting Actuator<Creating the Ball Shooter Shooting Actuator>`: A custom discrete :cpp:class:`Actuator <UActuator>` that allows the agent to shoot the ball.
-  :localref:`Discrete Rotation Actuator<Creating the Ball Shooter Discrete Rotation Actuator>`: A custom discrete :cpp:class:`Actuator <UActuator>` that allows the agent to rotate.
-  :localref:`Trainer blueprint<Creating the Trainer>`: A subclass of :cpp:class:`BlueprintTrainer <ABlueprintTrainer>`, which includes the logic to compute the :cpp:func:`reward <ABlueprintTrainer::ComputeReward>` and :cpp:func:`status <ABlueprintTrainer::ComputeReward>` of the training, as well as :cpp:class:`Sensors <USensor>` :cpp:class:`Actuators <UActuator>`.
-  :localref:`Environment definition<Creating the Environment Definition>`: A subclass of :cpp:class:`BlueprintStaticScholaEnvironment <ABlueprintStaticScholaEnvironment>`, which includes the logic of :cpp:func:`initializing<ABlueprintStaticScholaEnvironment::InitializeEnvironment>` and :cpp:func:`resetting<ABlueprintStaticScholaEnvironment::ResetEnvironment>` the environment between different episodes of training.  
-  :localref:`Registering the agent<Registering the Agent>`: Connect the agent to the environment definition and trainer.  


Initial Setup  
-------------  
  
1. Create a new blank project with a desired name and location.  
2. Install the Schola plugin to the project using the :doc:`/guides/setup_schola` guide.  
3. Go to Edit → Project Settings, and scroll down to find Schola. 

   .. note::
      
      If you don't see Schola in the Project Settings, please check whether Schola is installed in Edit → Plugins Menu. Please refer to the :doc:`/guides/setup_schola` guide for more information.
         
      .. image:: /_static/guides/example_one/plugin_menu.png
         :width: 450

4. For :cpp:var:`Gym Connector Class<UScholaManagerSubsystemSettings::GymConnectorClass>`, select :cpp:class:`Python Gym Connector <UPythonGymConnector>`  

.. image:: /_static/guides/example_two/create_blank_project.png  

.. image:: /_static/guides/example_one/schola_setting.png
  
Creating the Map
----------------
  
1. Create a shooting range with a floor and four walls in the map.  
2. For the walls, in ``Details`` → ``Tags``, add a new element, and set the value to ``wall``. This tag is used by the :cpp:class:`RayCastObserver <URayCastObserver>` to detect different objects.

.. image:: /_static/guides/example_two/BallShooterWallTags.png

.. image:: /_static/guides/example_two/BallShooterMap.png

Creating the Ball
-----------------
The Ball class is the projectile that the agent shoots. The ball is spawned by the agent when it takes the shooting action and is destroyed upon hitting a wall or target.

1. Create a new Blueprint Class with parent class `Actor <https://dev.epicgames.com/documentation/en-us/unreal-engine/actors>`_, and name it ``BallShooterBall``.
2. Add a Sphere `Static Mesh Component <https://dev.epicgames.com/documentation/en-us/unreal-engine/static-mesh-component>`_ to the blueprint, and optionally select a good-looking material.

   1. Enable ``Details`` → ``Physics`` → ``Simulate Physics``.
   2. Enable ``Details`` → ``Collision`` → ``Simulation Generates Hit Events``.
   3. Enable ``Details`` → ``Collision`` → ``Generate Overlap Events``.
   4. Set ``Details`` → ``Collision`` → ``Collision Presets`` to ``Custom``.
   5. Set ``Details`` → ``Collision`` → ``Collision Presets`` → ``Collision Enabled`` to ``Probe Only``. This prevents the ball from blocking the agent's :cpp:class:`Ray Cast Observer <URayCastObserver>` vision.
   
3. Add a Sphere `Collision Component <https://dev.epicgames.com/documentation/en-us/unreal-engine/shape-components>`_, making it slightly larger than the Sphere.
4. Scale the ``DefaultSceneRoot`` to 0.5x0.5x0.5.

.. image:: /_static/guides/example_two/BallShooterBall.config.png

Creating the Target 
-------------------

The target is the object that the agent will shoot. The target moves randomly around the map and is destroyed when hit three times by a ball. The Event Tick will apply a random force to the target to move it around the map. The ``OnTakeAnyDamage_Event`` will be triggered when hit by a ball, adjust the target's hitpoint, and destroy the target when the hitpoint reaches zero.

#. Create a new Blueprint Class with parent class `Actor <https://dev.epicgames.com/documentation/en-us/unreal-engine/actors>`_, and name it ``BallShooterTarget``.
#. Add a Sphere `Static Mesh Component <https://dev.epicgames.com/documentation/en-us/unreal-engine/static-mesh-component>`_ to the blueprint, and optionally select a good-looking material.

   1. Enable ``Details`` → ``Physics`` → ``Simulate Physics``.
   2. Enable ``Details`` → ``Collision`` → ``Simulation Generates Hit Events``.
   3. Enable ``Details`` → ``Collision`` → ``Generate Overlap Events``.
   4. Set ``Details`` → ``Collision`` → ``Collision Presets`` to ``PhysicsActor``.


#. Add a Sphere `Collision Component <https://dev.epicgames.com/documentation/en-us/unreal-engine/shape-components>`_, making it slightly larger than the Sphere.
#. Scale the ``DefaultSceneRoot`` to 3x3x3.
#. Add a new boolean variable. Name it ``isHit``. It stores whether the agent is hit by a ball in the current step.
#. Add a new Transform variable. Name it ``initialTransform``. It stores the initial transform of the target when the episode starts.
#. Add a new integer variable. Name it ``hitPoint``, and set the default value to 3. It stores the number of times the target is hit by a ball. The target will be destroyed when the hitpoint reaches zero. 
#. Add a new float variable. Name it ``forceMagnitude``, and  set the default value to 50. It stores the magnitude of the random force applied to the target on each tick.
#. Create a new function ``teleportToRandomLocation`` as shown below, and set the `Make Vector <https://dev.epicgames.com/documentation/en-us/unreal-engine/BlueprintAPI/Math/Vector/MakeVector>`_ node's random ranges to the range of the shooting range. This function will teleport the target to a random location within the shooting range.
#. Set the Event Graph as shown below.

   1. The `Event Begin Play <https://dev.epicgames.com/documentation/en-us/unreal-engine/BlueprintAPI/AddEvent/EventBeginPlay>`_ will save the initial transform of the target and bind the ``OnTakeAnyDamage_Event`` once.
   2. The ``OnTakeAnyDamage_Event`` will be triggered when hit by a ball, adjust the Target's hitpoint, and destroy the target when the hitpoint reaches zero. 
   3. The `Event Tick <https://dev.epicgames.com/documentation/en-us/unreal-engine/BlueprintAPI/AddEvent/EventTick>`_ will apply a random force to the target to move it around the shooting range. 

#. In ``Details`` → ``Tags``, add a new element, and set the value to ``target``. This tag is used by the :cpp:class:`RayCastObserver <URayCastObserver>` to detect different objects.

.. image:: /_static/guides/example_two/BallShooterTargetTags.png

.. image:: /_static/guides/example_two/BallShooterTarget.config.png

.. blueprint-file:: example_two/BallShooterTarget.bp
   :heading: BallShooterTarget > Event Graph
   :imagefallback: /_static/guides/example_two/BallShooterTarget.png
   :height: 500
   :zoom: -4

.. blueprint-file:: example_two/BallShooterTarget.teleportToRandomLocation.bp
   :heading: BallShooterTarget > teleportToRandomLocation
   :imagefallback: /_static/guides/example_two/BallShooterTarget.teleportToRandomLocation.png
   :height: 350
   :zoom: -4

Creating the Agent
------------------

#. Create a new Blueprint Class with parent class `Pawn <https://dev.epicgames.com/documentation/en-us/unreal-engine/pawn-in-unreal-engine>`_, and name it ``BallShooterAgent``.  
#. Add any desired `static meshes <https://dev.epicgames.com/documentation/en-us/unreal-engine/BlueprintAPI/StaticMesh>`_ as the agent’s body, and optionally select good-looking materials.  
#. Add an `Arrow Component <https://dev.epicgames.com/documentation/en-us/unreal-engine/API/Runtime/Engine/Components/UArrowComponent>`_, and set it as the agent’s forward direction. Name it ``Projectile Indicator``.
#. Save and close the blueprint, and place a ``BallShooterAgent`` at the center of the map.  

.. image:: /_static/guides/example_two/BallShooterMap2.png


Creating the Ball Shooter Shooting Actuator
-------------------------------------------

There are a variety of built-in actuator classes available in Schola, such as the :cpp:class:`TeleportActuator<UTeleportActuator>` and :cpp:class:`MovementInputActuator<UMovementInputActuator>`. However, some games may require custom actuators. In this example, we will create a custom :cpp:class:`BlueprintDiscreteActuator<UBlueprintDiscreteActuator>` (subclass of :cpp:class:`DiscreteActuator<UDiscreteActuator>`) to shoot the ball. This actuator has two possible actions: shoot a ball or do nothing. The :cpp:func:`~UBlueprintDiscreteActuator::GetActionSpace` function will return the action space, and the :cpp:func:`~UBlueprintDiscreteActuator::TakeAction` function will take the action. We will also create two helper functions, ``getInitialVelocity()`` and ``getInitialLocation()``, to get the initial velocity and location for spawning the ball.

.. note::
   A :cpp:class:`BinaryActuator<UBinaryActuator>` can also be used here instead of the :cpp:class:`DiscreteActuator<UDiscreteActuator>`. 


#. Create a new Blueprint Class with parent class :cpp:class:`BlueprintDiscreteActuator<UBlueprintDiscreteActuator>`, and name it ``BallShooterShootingActuator``.  
#. Add a new float variable. Name it ``ballSpawnSpeed``, and set the default value to 2000. This stores the speed of the ball when shot.
#. Add a new `Rotator <https://dev.epicgames.com/documentation/en-us/unreal-engine/BlueprintAPI/Math/Rotator>`_ variable. Name it ``projectileSpawnDirection``. This stores the direction in which the ball will be spawned. Adjust the values to ensure the ball is spawned in the correct direction.
#. Add a new float variable. Name it ``ballSpawnOffset``. This stores the offset from the agent's location where the ball will be spawned. Set the default value to 200, and adjust if necessary to ensure the ball is spawned in front of, not inside the agent.
#. Add a new integer variable. Name it ``countOfBallsShot``. It stores the number of balls shot by the agent in the current time step.
#. Add a new  `Actor <https://dev.epicgames.com/documentation/en-us/unreal-engine/actors>`_ variable. Name it ``Agent``. This stores the agent that owns the actuator. 
#. Convert the function :cpp:func:`~UBlueprintDiscreteActuator::TakeAction` into an `event <https://dev.epicgames.com/documentation/en-us/unreal-engine/events-in-unreal-engine>`_. This allows us to bind the ``Ball Hit Event`` to the spawned ball.
#. Set the ``getInitialVelocity()``, ``getInitialLocation()``, :cpp:func:`~UBlueprintDiscreteActuator::GetActionSpace`, and :cpp:func:`~UBlueprintDiscreteActuator::TakeAction` blueprints as shown below. 

.. blueprint-file:: example_two/BallShooterShootingActuator.bp
   :heading: BallShooterShootingActuator > Event Graph
   :imagefallback: /_static/guides/example_two/BallShooterShootingActuator.png
   :height: 400
   :zoom: -5

.. blueprint-file:: example_two/BallShooterShootingActuator.GetActionSpace.bp
   :heading: BallShooterShootingActuator > GetActionSpace
   :imagefallback: /_static/guides/example_two/BallShooterShootingActuator.GetActionSpace.png
   :height: 300
   :zoom: -1

.. blueprint-file:: example_two/BallShooterShootingActuator.getInitialVelocity.bp
   :heading: BallShooterShootingActuator > getInitialVelocity
   :imagefallback: /_static/guides/example_two/BallShooterShootingActuator.getInitialVelocity.png
   :height: 400
   :zoom: -3

.. blueprint-file:: example_two/BallShooterShootingActuator.getInitialLocation.bp
   :heading: BallShooterShootingActuator > getInitialLocation
   :imagefallback: /_static/guides/example_two/BallShooterShootingActuator.getInitialLocation.png
   :height: 400
   :zoom: -3

Creating the Ball Shooter Discrete Rotation Actuator
----------------------------------------------------

Although the :cpp:class:`RotationActuator<URotationActuator>` exists in Schola and can be used to rotate agents continuously, we will create another custom :cpp:class:`BlueprintDiscreteActuator<UBlueprintDiscreteActuator>` (subclass of :cpp:class:`DiscreteActuator<UDiscreteActuator>`) to rotate the agent. This actuator has three possible actions: rotate left, rotate right, or do nothing.

.. note:: 
   Mixing discrete and continuous actuators in the same agent should be avoided. The `stable-baseline3 <https://stable-baselines3.readthedocs.io/en/master/>`_ library and most algorithms in general do not support mixing discrete and continuous action spaces. Although some workarounds may exist, mixing may cause bugs or reduce training performance. Conversely, mixing discrete and continuous observers is completely supported.

#. Create a new Blueprint Class with parent class :cpp:class:`BlueprintDiscreteActuator<UBlueprintDiscreteActuator>`, and name it ``BallShooterDiscreteRotationActuator``.  
#. Add a new float variable. Name it ``rotationMagnitude``, and set the default value to 2. This stores the magnitude of the rotation when the agent rotates.
#. Set the :cpp:func:`~UBlueprintDiscreteActuator::GetActionSpace` and :cpp:func:`~UBlueprintDiscreteActuator::TakeAction` blueprints as shown below. 

.. blueprint-file:: example_two/BallShooterDiscreteRotationActuator.GetActionSpace.bp
   :heading: BallShooterDiscreteRotationActuator > GetActionSpace
   :imagefallback: /_static/guides/example_two/BallShooterDiscreteRotationActuator.GetActionSpace.png
   :height: 300
   :zoom: -1

.. blueprint-file:: example_two/BallShooterDiscreteRotationActuator.TakeAction.bp
   :heading: BallShooterDiscreteRotationActuator > TakeAction
   :imagefallback: /_static/guides/example_two/BallShooterDiscreteRotationActuator.TakeAction.png
   :height: 400
   :zoom: -4

Creating the Trainer  
--------------------

To train an agent in Schola, the agent must be controlled by an :cpp:class:`AbstractTrainer<AAbstractTrainer>`, which defines the :cpp:func:`~ABlueprintTrainer::ComputeReward` and :cpp:func:`~ABlueprintTrainer::ComputeStatus` functions.
In this tutorial, we will be creating an :cpp:class:`BlueprintTrainer <ABlueprintTrainer>` (subclass of :cpp:class:`AbstractTrainer<AAbstractTrainer>`).

#. Create a new Blueprint Class with parent class :cpp:class:`BlueprintTrainer <ABlueprintTrainer>`, and name it ``BallShooterTrainer``.  
#. Add a new integer variable. Name it ``maxNumberOfHitsPerEpisode``. It stores the maximum number of times the agent can hit the target in one episode, which is the number of targets multiplied by the number of hitpoints for each target. It is set by the :localref:`Environment Definition <Creating the Environment Definition>` blueprint.
#. Add a new integer variable. Name it ``numOfHitsThisEpisode``. It stores the number of times the agent has hit the target in the current episode. It is used to determine when the episode ends.
#. Add a new integer variable. Name it ``numOfTargetHits``. It stores the number of times the agent has hit the target in the current step.
#. Add an :cpp:class:`Actuator <UActuator>` component, and set the ``Details`` → :cpp:class:`Actuator Component <UActuatorComponent>` → :cpp:class:`Actuator <UActuator>` to ``BallShooterShootingActuator``
#. Set the Event Graph as shown below. This binds the ``On Ball Hit`` event to any balls spawned by the agent's actuator, allowing the trainer to detect when the agent hits or misses the target.

.. blueprint-file:: example_two/BallShooterTrainer.bp
   :heading: BallShooterTrainer > Event Graph
   :imagefallback: /_static/guides/example_two/BallShooterTrainer.png
   :height: 400
   :zoom: -4

.. _attaching-actuators-and-observers:

Attaching Actuators and Observers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Unlike the :doc:`Example 1 <example_one>`, actuators and observers will not be attached to the agent blueprint. Instead, they will be attached in the :cpp:class:`Trainer <ABlueprintTrainer>` blueprint. 
This approach simplifies passing variables, as the  :cpp:class:`Trainer's <ABlueprintTrainer>` :cpp:func:`~ABlueprintTrainer::ComputeReward` and :cpp:func:`~ABlueprintTrainer::ComputeStatus` logic rely on variables from the ``BallShooterDiscreteRotationActuator``. 

.. note::

   :cpp:class:`Actuator <UActuator>` objects can be attached in three ways:
   
   1. Attaching an :cpp:class:`ActuatorComponent <UActuatorComponent>` to the agent, which can contain an :cpp:class:`Actuator<UActuator>` object.
   2. Attaching an :cpp:class:`ActuatorComponent <UActuatorComponent>` component to the :cpp:class:`BlueprintTrainer <ABlueprintTrainer>`, which can contain an :cpp:class:`Actuator<UActuator>` object.
   3. Adding directly in the :cpp:var:`~AAbstractTrainer::Actuators` arrays in the :cpp:class:`BlueprintTrainer <ABlueprintTrainer>`.

Attaching the Ball Shooter Shooting Actuator
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

#. Add an :cpp:class:`Actuator <UActuator>` component.  
#. In ``Details`` → ``Actuator Component`` → ``Actuator``, select ``BallShooterDiscreteRotationActuator``.  

.. image:: /_static/guides/example_two/BallShooterShootingActuatorTrainer.png


Attaching the Ball Shooter Discrete Rotation Actuator
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

#. Add an :cpp:class:`Actuator <UActuator>` component.  
#. In ``Details`` → ``Actuator Component`` → ``Actuator``, select ``BallShooterDiscreteRotationActuator``.  

.. image:: /_static/guides/example_two/BallShooterRotationActuatorTrainer.png

Attaching the Ray Cast Observer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

#. Add a :cpp:class:`Sensor <USensor>` component.  
#. In ``Details`` → ``Sensor`` → ``Observer``, select :cpp:class:`Ray Cast Observer <URayCastObserver>`.  
#. In ``Details`` → ``Sensor`` → ``Observer``→ ``Sensor properties`` → :cpp:var:`~URayCastObserver::NumRays`, enter 10.  
#. In ``Details`` → ``Sensor`` → ``Observer``→ ``Sensor properties`` → :cpp:var:`~URayCastObserver::RayDegrees`, enter 120.  
#. In ``Details`` → ``Sensor`` → ``Observer``→ ``Sensor properties``, check the :cpp:var:`DrawDebugLines <URayCastObserver::bDrawDebugLines>` box.  
#. In ``Details`` → ``Sensor`` → ``Observer``→ ``Sensor properties`` → :cpp:var:`~URayCastObserver::TrackedTags`, add a new element, and set the tag to ``target``.  

.. image:: /_static/guides/example_two/BallShooterRayCast.png


Define the Reward Function
~~~~~~~~~~~~~~~~~~~~~~~~~~

In this tutorial, we give a reward of 1 for hitting a target and a penalty of -0.01 for missing the target. Additionally, we give a small penalty of -0.05 for each step the agent takes, to encourage the agent to destroy all targets and end the episode as quickly as possible. The per-step reward is computed as ``(1.01*numOfTargetHits - 0.01*countOfBallsShot) - 0.05`` 

1. Add a new float variable. Name it ``reward``. It stores the reward for the current step.
2. Set the :cpp:func:`~ABlueprintTrainer::ComputeReward` function as shown below.  

.. blueprint-file:: example_two/BallShooterTrainer.ComputeReward.bp
   :heading: BallShooterTrainer > ComputeReward
   :imagefallback: /_static/guides/example_two/BallShooterTrainer.ComputeReward.png
   :height: 300
   :zoom: -5

Define the Status Function  
~~~~~~~~~~~~~~~~~~~~~~~~~~  
  
There are three possible statuses for each time step:  
  
1. **Running**: The episode is still ongoing, and the agent continues interacting with the environment.  
2. **Completed**: The agent has successfully reached a terminal state, completing the episode.  
3. **Truncated**: The episode has been forcefully ended, often due to external limits like time steps or manual intervention, without reaching the terminal state.  
  
In this tutorial, the terminal state is reached when the agent destroys all targets, which is when the ``numOfTargetHits`` is equal to the ``maxNumberOfHitsPerEpisode``. We also set a max step to prevent an episode from running indefinitely. 

1. Add a new integer variable. Name it ``maxStep``, and set the default value to 1000. This means an episode is truncated if it reaches 1000 time steps without completing. You may adjust this number if you want to allow longer or shorter episodes due to factors such as the size of the environment or the speed of the agent.
2. Set the :cpp:func:`~ABlueprintTrainer::ComputeStatus` as shown below.  

.. blueprint-file:: example_two/BallShooterTrainer.ComputeStatus.bp
   :heading: BallShooterTrainer > ComputeStatus
   :imagefallback: /_static/guides/example_two/BallShooterTrainer.ComputeStatus.png
   :height: 350
   :zoom: -4

.. note::

   The :cpp:var:`~AAbstractTrainer::Step` variable is a part of the :cpp:class:`BlueprintTrainer <ABlueprintTrainer>` and it tracks the current number of steps since the last :cpp:func:`~ABlueprintStaticScholaEnvironment::ResetEnvironment` call.


Creating the Environment Definition
-----------------------------------

To train an agent in Schola, the game must have an :cpp:class:`StaticScholaEnvironment<AStaticScholaEnvironment>` Unreal object, which contains the agent and logic for initializing or resetting the game environment. 
In this tutorial, we will be creating an :cpp:class:`Blueprint Environment<ABlueprintStaticScholaEnvironment>` (subclass of :cpp:class:`StaticScholaEnvironment<AStaticScholaEnvironment>`) as the Environment.
The :cpp:func:`~ABlueprintStaticScholaEnvironment::InitializeEnvironment` function is called at the start of the game, and sets the initial state of the environment.
In this tutorial, we save the initial transform (position and rotation) of the agent.
The :cpp:func:`~ABlueprintStaticScholaEnvironment::ResetEnvironment` function is called before every new episode. In this tutorial, we reset the agent to its initial transform, clean up any leftover balls and targets, spawn three new targets, calculate the ``TotalHitPoints`` for the episode, and reset the variables in the trainer.

1. Create a new Blueprint Class with parent class :cpp:class:`BlueprintStaticScholaEnvironment <ABlueprintStaticScholaEnvironment>`, and name it ``BallShooterEnvironment``.  
2. Add a new variable named ``agentArray`` of type `Pawn (Object Reference) <https://dev.epicgames.com/documentation/en-us/unreal-engine/pawn-in-unreal-engine>`_ array. This keeps track of registered agents belonging to this environment definition.
  
   1. Make this variable publicly editable (by clicking on the eye icon to toggle the visibility).  
  
3. Add a new `Transform <https://dev.epicgames.com/documentation/en-us/unreal-engine/BlueprintAPI/Math/Transform>`_ variable named ``agentInitialLocation``. This is for storing the initial position and rotation of the agent, so it can be restored upon reset.  
4. Add a new integer variable named ``numberOfTargets``, and set the default value to 3. This stores the number of targets to spawn in the environment.
5. Add a new integer variable named ``totalHitPoints``. This stores the total number of hit points for the episode, which is the number of targets multiplied by the number of hitpoints for each target.
6. Add a new variable named ``Targets`` of type ``Ball Shooter Target (Object Reference)`` array. This stores the spawned targets in the environment.
7. Create functions ``saveAgentInitialTransform`` and ``placeAgentToInitialTransform`` as shown below. This saves the initial transform of the agent and places the agent to its initial transform when the episode starts.
8. Set the Event Graph and :cpp:func:`~ABlueprintStaticScholaEnvironment::RegisterAgents` function as shown below. 
9. Save and close the blueprint, and place a ``BallShooterEnvironment``  anywhere in the map. The location does not matter.  

.. blueprint-file:: example_two/BallShooterEnvironment.saveAgentInitialTransform.bp
   :heading: BallShooterEnvironment > saveAgentInitialTransform
   :imagefallback: /_static/guides/example_two/BallShooterEnvironment.saveAgentInitialTransform.png
   :height: 250
   :zoom: -2

.. blueprint-file:: example_two/BallShooterEnvironment.placeAgentToInitialTransform.bp
   :heading: BallShooterEnvironment > placeAgentToInitialTransform
   :imagefallback: /_static/guides/example_two/BallShooterEnvironment.placeAgentToInitialTransform.png
   :height: 250
   :zoom: -2

.. blueprint-file:: example_two/BallShooterEnvironment1.bp
   :heading: BallShooterEnvironment > Event Graph (part 1)
   :imagefallback: /_static/guides/example_two/BallShooterEnvironment.png
   :height: 600
   :zoom: -5

.. blueprint-file:: example_two/BallShooterEnvironment2.bp
   :heading: BallShooterEnvironment > Event Graph (part 2)
   :imagefallback: /_static/guides/example_two/BallShooterEnvironment2.png
   :height: 600
   :zoom: -5


.. blueprint-file:: example_two/BallShooterEnvironment.RegisterAgents.bp
   :heading: BallShooterEnvironment > RegisterAgents
   :imagefallback: /_static/guides/example_two/BallShooterEnvironment.RegisterAgents.png
   :height: 250
   :zoom: -1

Registering the Agent
---------------------

1. Click on the ``BallShooterEnvironment``  in the map.  
  
   1. Go to ``Details`` panel → ``Default`` → ``Agent Array``.  
   2. Add a new element.  
   3. Select ``BallShooterAgent`` in the drop-down menu.  
  
      .. image:: /_static/guides/example_two/ball_shooter_environment_include_pawn.png  
         :width: 400  
  
2. Open the ``BallShooterAgent`` class in the blueprint editor.  
  
   1. Go to Details Panel.  
   2. Search for `AIController <https://dev.epicgames.com/documentation/en-us/unreal-engine/ai-controllers-in-unreal-engine>`_.  
   3. In the drop-down, select ``BallShooterTrainer`` .  
  
      .. image:: /_static/guides/example_two/ball_shooter_aicontroller.png  
         :width: 400  

Starting Training   
-----------------  

We will train the agent using the `Proximal Policy Optimization (PPO) <https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html>`_ algorithm for 100,000 steps.
The following two methods run the same training. Running from the terminal may be more convenient for hyperparameter tuning, while running from the Unreal Editor may be more convenient when editing the game.

.. tabs::  
  
   .. group-tab:: Run from terminal  
  
      1. Run the game in Unreal Engine (by clicking the green triangle).  
      2. Open a terminal or command prompt, and run the following Python script:  
  
      .. code-block:: bash  
  
         schola-sb3 -p 8000 -t 100000 PPO

      .. note::
         
         To run with RLlib, use the ``schola-rllib`` command instead of ``schola-sb3``.

         .. code-block:: bash
               
               schola-rllib -p 8000 -t 100000
  

   .. group-tab:: Run from Unreal Editor  

      Schola can also run the training script directly from the Unreal Editor. 
          
      1. Go to ``Edit`` → ``Project Settings``, and scroll down to find Schola.
      2. Check the :cpp:class:`Run Script on Play <UScholaManagerSubsystemSettings>` box.  
      3. Change :cpp:var:`~UScholaManagerSubsystemSettings::ScriptSettings` → :cpp:var:`~FScriptSettings::SB3Settings` → :cpp:var:`~FSB3TrainingSettings::Timesteps` to 100000.
      4. Run the game in Unreal Engine (by clicking the green triangle).  

      .. note::
   
         By default, Schola runs the ``python`` command when launching Python. If you have Python installed differently, such as ``python3.9``, or ``/usr/bin/python3``, 
         please change

         1. :cpp:var:`~UScholaManagerSubsystemSettings::ScriptSettings` → :cpp:var:`~FScriptSettings::EnvType` to ``Custom Python Path``.
         2. :cpp:var:`~UScholaManagerSubsystemSettings::ScriptSettings` → :cpp:var:`~FScriptSettings::CustomPythonPath` to your Python path or alias.
         

      .. image:: /_static/guides/example_two/running_from_editor.png   
      
      .. note::
   
         To run with RLlib, make the following changes the ``Edit`` → ``Project Settings``:

         #. Change :cpp:var:`~UScholaManagerSubsystemSettings::ScriptSettings` → :cpp:var:`~FScriptSettings::PythonScriptType` to ``Builtin RLlib Training Script``.
         #. Change :cpp:var:`~UScholaManagerSubsystemSettings::ScriptSettings` → :cpp:var:`~FScriptSettings::RLlibSettings` → :cpp:var:`~FRLlibTrainingSettings::Timesteps` to 100000.

Enabling TensorBoard
~~~~~~~~~~~~~~~~~~~~

TensorBoard is a visualization tool provided by TensorFlow that allows you to track and visualize metrics such as loss and reward during training. 

.. tabs::  
  
   .. group-tab:: Run from terminal  

      Add the ``--enable-tensorboard`` flag to the command to enable TensorBoard. The ``--log-dir`` flag sets the directory where the logs are saved.

      .. code-block:: bash  
  
         schola-sb3 -p 8000 -t 100000 --enable-tensorboard --log-dir experiment_ball_shooter PPO  

      .. note:: 

         Running with RLlib using ``schola-rllib`` already enables TensorBoard by default.
         
   .. group-tab:: Run from Unreal Editor  

      Schola can also run the training script directly from the Unreal Editor. 
          
      1. Go to ``Edit`` → ``Project Settings``, and scroll down to find Schola.
      2. Check the :cpp:var:`~UScholaManagerSubsystemSettings::ScriptSettings` → :cpp:var:`~FScriptSettings::SB3Settings` → :cpp:var:`~FSB3TrainingSettings::LoggingSettings` → :cpp:var:`SaveTBLogs<FSB3LoggingSettings::bSaveTBLogs>` box.
      3. Set the :cpp:var:`~UScholaManagerSubsystemSettings::ScriptSettings` → :cpp:var:`~FScriptSettings::SB3Settings` → :cpp:var:`~FSB3TrainingSettings::LoggingSettings` → :cpp:var:`SaveTBLogs<FSB3LoggingSettings::LogDir>` to ``experiment_ball_shooter`` or another location for the logs to be saved.
      4. Run the game in Unreal Engine (by clicking the green triangle).           

      .. image:: /_static/guides/example_two/running_from_editor_with_tensorboard.png   

      .. note:: 

         Running with RLlib already enables TensorBoard by default.

After training, you can view the training progress in TensorBoard by running the following command in the terminal or command prompt. Make sure to first `install TensorBoard <https://pypi.org/project/tensorboard>`_, and set the ``--logdir`` to the directory where the logs are saved.

.. code-block:: bash

   tensorboard --logdir experiment_ball_shooter/PPO_1

.. note::

   Logs for subsequent ``schola-sb3`` runs will be in ``PPO_2``, ``PPO_3``, etc.

.. note:: 

   If you are running with RLlib, the logs will be saved in the ``ckpt/PPO_timestamp`` directory.

.. image:: /_static/guides/example_two/ball_shooter_tensorboard.png  
