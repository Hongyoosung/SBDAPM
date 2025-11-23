Building Tag
============

In this tutorial, we create a multi-agent environment where the agents are trained to play a 3v1 game of tag. Specifically, we create one runner agent which tries to avoid being caught and three tagger agents with the goal of catching the runner. The agents can move forward, left and right and can sense both their surrounding objects, as well as the locations of other agents.

.. image:: /_static/guides/example_three/Tag.gif
   :align: center  
   :scale: 150  

The Structure of the Environment in Unreal Engine
-------------------------------------------------

To build the game (called environment hereafter), we need to create the following in our Unreal Engine project:  

-  :localref:`Direction and Distance Observer<Creating the Custom Direction and Distance Observer>`: A custom :cpp:class:`BlueprintBoxObserver<UBlueprintBoxObserver>` that allows the taggers to observe the direction and distance of other agents.
-  :localref:`Agent blueprint<Creating the Agent>`: A subclass of `Character <https://dev.epicgames.com/documentation/en-us/unreal-engine/characters-in-unreal-engine>`_, which includes the shape and appearance of the agent.
-  :localref:`Trainer blueprint<Creating the Trainer>`: A subclass of :cpp:class:`BlueprintTrainer <ABlueprintTrainer>`, which includes the logic to compute the :cpp:func:`reward <ABlueprintTrainer::ComputeReward>` and :cpp:func:`status <ABlueprintTrainer::ComputeReward>` of the training.
-  :localref:`Environment definition<Creating the Environment Definition>`: A subclass of :cpp:class:`BlueprintStaticScholaEnvironment <ABlueprintStaticScholaEnvironment>`, which includes the logic of :cpp:func:`initializing<ABlueprintStaticScholaEnvironment::InitializeEnvironment>` the environment before training starts and :cpp:func:`resetting<ABlueprintStaticScholaEnvironment::ResetEnvironment>` the environment between different episodes of training.  
-  :localref:`Map<Creating the Map>`: The game map includes the floor, four walls, agents, and the environment.  
-  :localref:`Registering the agents<Registering the Agents>`: Connect the agents to the environment and their respective trainers. 

Initial Setup  
-------------
Please refer to the :ref:`Schola Initial Setup<setup_unreal_engine>` section to set up the Unreal Engine project and Schola plugin.  

Creating the Custom Direction and Distance Observer
---------------------------------------------------

There are a variety of built-in observer classes available in Schola, such as the :cpp:class:`RotationObserver<URotationObserver>` and :cpp:class:`RayCastObserver<URayCastObserver>`. Custom observers are needed when we need specific observations not covered by the built-in observers. In this example, we will create a custom :cpp:class:`BlueprintBoxObserver<UBlueprintBoxObserver>` (subclass of :cpp:class:`BoxObserver<UBoxObserver>`) to allow taggers to observe the direction and distance of other agents relative to the current agent in the game. It will return the distance normalized by the environment size and the direction as a unit vector. The :cpp:func:`~UBlueprintBoxObserver::GetObservationSpace` function will return the observation space, and the :cpp:func:`~UBlueprintBoxObserver::CollectObservations` function will collect and return the observations.

#. Create a new Blueprint Class with parent class :cpp:class:`BlueprintBoxObserver<UBlueprintBoxObserver>`, and name it ``DirectionDistanceObserver``.  
#. Add a new integer variable. Name it ``EnvSize``, and set the default value to 5000. This stores the maximum possible distance between two agents within the environment.
#. Add a new `Actor <https://dev.epicgames.com/documentation/en-us/unreal-engine/actors-in-unreal-engine>`_ variable. Name it ``Target``. This stores the target agent that the observer will track.
#. Set the :cpp:func:`~UBlueprintBoxObserver::GetObservationSpace` and :cpp:func:`~UBlueprintBoxObserver::CollectObservations` blueprints as shown below. 

.. blueprint-file:: example_three/DirectionDistanceObserver.CollectObservations.bp
   :heading: DirectionDistanceObserver > CollectObservations
   :imagefallback: /_static/guides/example_three/DirectionDistanceObserver.CollectObservations.png
   :height: 350
   :zoom: -4

.. blueprint-file:: example_three/DirectionDistanceObserver.GetObservationSpace.bp
   :heading: DirectionDistanceObserver > GetObservationSpace
   :imagefallback: /_static/guides/example_three/DirectionDistanceObserver.GetObservationSpace.png
   :height: 250
   :zoom: -3

Creating the Agent
------------------

Creating the Tagger Class
~~~~~~~~~~~~~~~~~~~~~~~~~

#. Create a new Blueprint Class with parent class `Character <https://dev.epicgames.com/documentation/en-us/unreal-engine/characters-in-unreal-engine>`_, and name it ``Tagger``.
#. Add any desired `static meshes <https://dev.epicgames.com/documentation/en-us/unreal-engine/BlueprintAPI/StaticMesh>`_ and `material <https://dev.epicgames.com/documentation/en-us/unreal-engine/unreal-engine-materials>`_ as the agent’s body. 
#. Set ``Details`` → ``Character Movement: Walking`` → ``Max Walk Speed`` to 520 cm/s.
#. Set ``Details`` → ``Character Movement (Rotation Settings)`` → ``Orient Rotation to Movement`` to true. This allows the agent to rotate using the :cpp:class:`Movement Input Actuator<UMovementInputActuator>`.
#. Set ``Details`` → ``Pawn`` → ``Use Controller Rotation Yaw`` to false. This allows the agent to rotate using the :cpp:class:`Movement Input Actuator<UMovementInputActuator>`.
#. In ``Details`` → ``Tags``, add a new tag, and set the value to ``Tagger``. This tag is used by the :cpp:class:`RayCastObserver <URayCastObserver>` to detect different objects.

.. image:: /_static/guides/example_three/taggerSettings.png

Attaching the Ray Cast Observer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

#. Add a :cpp:class:`Sensor <USensor>` component.  
#. In ``Details`` → ``Sensor`` → ``Observer``, select :cpp:class:`Ray Cast Observer <URayCastObserver>`.  
#. Set ``Details`` → ``Sensor`` → ``Observer`` → ``Sensor properties`` → :cpp:var:`~URayCastObserver::NumRays` to 36.  
#. Set ``Details`` → ``Sensor`` → ``Observer`` → ``Sensor properties`` → :cpp:var:`~URayCastObserver::RayDegrees` to 360.  
#. Set ``Details`` → ``Sensor`` → ``Observer`` → ``Sensor properties`` → :cpp:var:`~URayCastObserver::RayLength` to 2048.  
#. In ``Details`` → ``Sensor`` → ``Observer`` → ``Sensor properties`` → :cpp:var:`~URayCastObserver::TrackedTags`, add two new elements and set the tags to ``Tagger`` and ``Runner``.  

.. note::

   For more information on attaching actuators and observers, please refer to the :ref:`Attaching Actuators and Observers Section of Example 2 <attaching-actuators-and-observers>`.

Attaching the Movement Input Actuator
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We will use two :cpp:class:`Movement Input Actuators <UMovementInputActuator>` to move the agent. One lateral axis actuator to steer, and one forward axis actuator to move the agent forward.

#. Add an :cpp:class:`Actuator <UActuator>` component, and name it ``ForwardAxisMovementInputActuator``
#. In ``Details`` → ``Actuator Component`` → ``Actuator``, select :cpp:class:`Movement Input Actuator <UMovementInputActuator>`.  
#. In ``Details`` → ``Actuator Component`` → ``Actuator`` → ``Actuator Settings``, uncheck :cpp:var:`HasYDimension <UMovementInputActuator::bHasYDimension>` and :cpp:var:`HasZDimension <UMovementInputActuator::bHasZDimension>`.  
#. Add an :cpp:class:`Actuator <UActuator>` component, and name it ``LateralAxisMovementInputActuator`` 
#. In ``Details`` → ``Actuator Component`` → ``Actuator``, select :cpp:class:`Movement Input Actuator <UMovementInputActuator>`.  
#. In ``Details`` → ``Actuator Component`` → ``Actuator`` → ``Actuator Settings``, uncheck :cpp:var:`HasXDimension <UMovementInputActuator::bHasXDimension>` and :cpp:var:`HasZDimension <UMovementInputActuator::bHasZDimension>`.  
#. In ``Details`` → ``Actuator Component`` → ``Actuator`` → ``Actuator Settings``, set :cpp:var:`~UMovementInputActuator::Minspeed` to -1.  

Attaching the Direction and Distance Observer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

#. Add three :cpp:class:`Sensor <USensor>` components, and name them ``Teammate Sensor 1``, ``Teammate Sensor 2``, and ``Runner Sensor``.  
#. For each sensor, in ``Details`` → ``Sensor`` → ``Observer``, select ``DirectionDistanceObserver``.
#. The ``Target`` variable of each sensor will be set in the :localref:`Registering the Agent` section.``

Creating the Runner Class
~~~~~~~~~~~~~~~~~~~~~~~~~

The runner is constructed similarly to the tagger but with some minor changes. Please repeat the steps in the :localref:`Creating the Tagger Class` section with the following changes:

#. Add the same :cpp:class:`RayCastObserver <URayCastObserver>` and :cpp:class:`MovementInputActuator <UMovementInputActuator>` to the runner class, but not the ``DirectionDistanceObserver``.
#. Set ``Details`` → ``Character Movement: Walking`` → ``Max Walk Speed`` to 490 cm/s. We will make the runner slower initially to make it easier for the tagger to catch the runner, so the tagger can learn to catch the runner at the beginning of the training. If the runner is as fast or faster than the tagger, the taggers may never catch the runner, preventing the taggers from learning. This can be manually increased during training as the tagger improves and can consistently catch the slower runner.
#. In ``Details`` → ``Tags``, add a new element, and set the value to ``Runner``. This tag is used by the :cpp:class:`RayCastObserver <URayCastObserver>` to detect different objects.


Creating the Trainer  
--------------------

We will create two :cpp:class:`BlueprintTrainers <ABlueprintTrainer>`, one for the tagger agent and one for the runner agent. 

Creating the Tagger Trainer
~~~~~~~~~~~~~~~~~~~~~~~~~~~

#. Create a new Blueprint Class with parent class :cpp:class:`BlueprintTrainer <ABlueprintTrainer>`, and name it ``TaggerTrainer``.  
#. Add a new boolean variable. Name it ``CaughtTarget``. It stores whether the tagger agent has caught the runner agent in the current step. It is set by the :localref:`Environment Definition <Creating the Environment Definition>` blueprint.
#. Add a new boolean variable. Name it ``HitWall``. It stores whether the tagger agent has hit a wall in the current step. It is set by the :localref:`Environment Definition <Creating the Environment Definition>` blueprint.
#. Add a new ``Tagger`` variable. Name it ``Agent``. It stores the pawn that the trainer controls.
#. Enable ``Details`` → ``Reinforcement Learning`` → :cpp:var:`~AAbstractTrainer::Name`, and set it to ``TaggerUnifiedPolicy`` or any string. This string determines the policy used during training so having all Taggers use the same name, makes all instances of Tagger Trainer share the same policy. Therefore the three tagger agents will train and use the same model.
#. Set ``Details`` → `Interaction Manager` → :cpp:var:`~AAbstractTrainer::DecisionRequestFrequency` to 1. This makes the agent decide an action at every step, allowing faster training. 
#. Set the Event Graph as shown below. 

.. note::

    By default, ``Details`` → ``Reinforcement Learning`` → :cpp:var:`~AAbstractTrainer::Name` is disabled, and every trainer will create a separate policy. When :cpp:var:`~AAbstractTrainer::Name` is enabled and set to any string, all trainers with this same name will share the same policy. This is useful when you want to train multiple agents with the same policy. This only works with frameworks supporting multi-agent training, such as `RLlib <https://docs.ray.io/en/latest/rllib/index.html>`_.

.. blueprint-file:: example_three/TaggerTrainer.bp
   :heading: TaggerTrainer > Event Graph
   :imagefallback: /_static/guides/example_three/TaggerTrainer.png
   :height: 300
   :zoom: -2


Define the Tagger Reward Function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We give a large one-time reward when the tagger agent catches the runner agent, and a small penalty of -0.015 when the tagger agent hits a wall. Additionally, we give a small penalty of -0.005 for each step the tagger agent takes, to encourage the agent to catch the runner agent as quickly as possible. The one-time reward is computed as ``10 - (0.0005 * DistanceFromRunner)``, where 10 is the maximum reward for catching the runner, and ``-0.0005*DistanceFromRunner`` decreases the reward as the tagger gets further from the runner to ensure taggers near the runner are rewarded more when the runner is caught. The two numbers are chosen based on our experience and can be adjusted as needed. The per-step reward is computed as ``-(0.015*HitWall) - 0.005``.

#. Set the :cpp:func:`~ABlueprintTrainer::ComputeReward` function as shown below.  

.. blueprint-file:: example_three/TaggerTrainer.ComputeReward.bp
   :heading: TaggerTrainer > ComputeReward
   :imagefallback: /_static/guides/example_three/TaggerTrainer.ComputeReward.png
   :height: 400
   :zoom: -3

Define the Tagger Status Function  
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For taggers, the terminal state is reached when the runner is caught. We also set a max step to prevent an episode from running indefinitely. For more information on the :cpp:var:`~AAbstractTrainer::Step` variable and :cpp:func:`~ABlueprintTrainer::ComputeStatus` function, please refer to :ref:`Example 1<maze_solver_status_function>`.

#. Add a new integer variable. Name it ``MaxSteps``, and set the default value to 2000. This stores the maximum number of steps an episode can run before ending. This may be set to a higher value if the tagger is unable to catch the runner within 2000 steps.
#. Set the :cpp:func:`~ABlueprintTrainer::ComputeStatus` as shown below.  

.. blueprint-file:: example_three/TaggerTrainer.ComputeStatus.bp
   :heading: TaggerTrainer > ComputeStatus
   :imagefallback: /_static/guides/example_three/TaggerTrainer.ComputeStatus.png
   :height: 300
   :zoom: -2

Creating the Runner Trainer
~~~~~~~~~~~~~~~~~~~~~~~~~~~

#. Create a new Blueprint Class with parent class :cpp:class:`BlueprintTrainer <ABlueprintTrainer>`, and name it ``RunnerTrainer``.  
#. Add a new boolean variable. Name it ``CaughtTarget``. It stores whether the tagger agent has caught the runner agent in the current step. It is set by the :localref:`Environment Definition <Creating the Environment Definition>` blueprint.
#. Set ``Details`` → `Interaction Manager` → :cpp:var:`~AAbstractTrainer::DecisionRequestFrequency` to 1. This makes the agent decide an action at every step, allowing smoother action. 

Define the Runner Reward Function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We give a large one-time penalty of -20 when the runner agent is caught and a small constant per-step reward of 0.01 to encourage the runner to survive as long as possible.

1. Set the :cpp:func:`~ABlueprintTrainer::ComputeReward` function as shown below.  

.. blueprint-file:: example_three/RunnerTrainer.ComputeReward.bp
   :heading: RunnerTrainer > ComputeReward
   :imagefallback: /_static/guides/example_three/RunnerTrainer.ComputeReward.png
   :height: 300
   :zoom: -3

Define the Runner Status Function  
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The runner has the same status function as the :localref:`Tagger Trainer <Creating the Tagger Trainer>`.

#. Add a new integer variable. Name it ``MaxSteps``, and set the default value to 2000. This stores the maximum number of steps an episode can run before ending. This may be set to a higher value if you find that during training the taggers are routinely unable to catch the runner before the episode ends.
#. Set the :cpp:func:`~ABlueprintTrainer::ComputeStatus` as shown below.  

.. blueprint-file:: example_three/RunnerTrainer.ComputeStatus.bp
   :heading: RunnerTrainer > ComputeStatus
   :imagefallback: /_static/guides/example_three/RunnerTrainer.ComputeStatus.png
   :height: 330
   :zoom: -3

Creating the Environment Definition
-----------------------------------

We will create a ``SetRunnerTagged`` function in the environment which notifies all the trainers when the runner is caught. The :cpp:func:`~ABlueprintStaticScholaEnvironment::InitializeEnvironment` binds a ``OnActorHit`` Event to each runner, that calls the ``SetRunnerTagged`` function when a runner comes into contact with a tagger. The :cpp:func:`~ABlueprintStaticScholaEnvironment::ResetEnvironment` function moves each agent to a random location and resets the variables in the trainer, at the end of each episode.

#. Create a new Blueprint Class with parent class :cpp:class:`BlueprintStaticScholaEnvironment <ABlueprintStaticScholaEnvironment>`, and name it ``TagEnvironment``.  
#. Add a new variable named ``Agents`` of type `Pawn (Object Reference) <https://dev.epicgames.com/documentation/en-us/unreal-engine/pawn-in-unreal-engine>`_ array, and make it publicly editable (by clicking on the eye icon to toggle the visibility). This keeps track of registered agents belonging to this environment definition. 
#. Create the ``SetRunnerTagged`` function as shown below.
#. Set the Event Graph and :cpp:func:`~ABlueprintStaticScholaEnvironment::RegisterAgents` function as shown below. 

.. blueprint-file:: example_three/TagEnvironment.SetRunnerTagged.bp
   :heading: TagEnvironment > SetRunnerTagged
   :imagefallback: /_static/guides/example_three/TagEnvironment.SetRunnerTagged.png
   :height: 300
   :zoom: -3

.. blueprint-file:: example_three/TagEnvironment.bp
   :heading: TagEnvironment > Event Graph
   :imagefallback: /_static/guides/example_three/TagEnvironment.png
   :height: 430
   :zoom: -5

.. blueprint-file:: example_three/TagEnvironment.RegisterAgents.bp
   :heading: TagEnvironment > RegisterAgents
   :imagefallback: /_static/guides/example_three/TagEnvironment.RegisterAgents.png
   :height: 220
   :zoom: -1

Creating the Map
----------------
  
#. Create a level with a floor and four walls.  
#. Add obstacles and decorations as desired.
#. Place a ``TagEnvironment``  anywhere in the map. The location does not matter.  
#. Place three ``Taggers`` near the centre of the map.
#. Place a ``Runner`` near the taggers.

Registering the Agents
----------------------

1. Select the ``TagEnvironment``  in the map.  
  
   1. Go to ``Details`` panel → ``Default`` → ``Agents``.  
   2. Add 4 new elements, and set the value to the four agents in the map.  
    
2. Open the ``Tagger`` class in the blueprint editor.  
  
   1. Go to Details Panel.  
   2. Search for `AIController <https://dev.epicgames.com/documentation/en-us/unreal-engine/ai-controllers-in-unreal-engine>`_.  
   3. In the drop-down, select ``TaggerTrainer`` .  
  
3. Open the ``Runner`` class in the blueprint editor.  
  
   1. Go to Details Panel.  
   2. Search for `AIController <https://dev.epicgames.com/documentation/en-us/unreal-engine/ai-controllers-in-unreal-engine>`_.  
   3. In the drop-down, select ``RunnerTrainer`` .  

4. Select a tagger in the map. 
   
   1. Go to Details Panel.  
   2. Select the ``Teammate Sensor 1`` component, set the ``Target`` to one of the other taggers, and repeat this for ``Teammate Sensor 2``.
   3. Select the ``Runner Sensor`` component, and set the ``Target`` to the runner.
   4. Repeat this for the other two taggers.

Starting Training   
-----------------  
 
We will train the agent using the `Proximal Policy Optimization (PPO) <https://docs.ray.io/en/latest/rllib/rllib-algorithms.html#ppo>`_ algorithm for 2,000,000 steps.
Since `SB3 <https://stable-baselines3.readthedocs.io/>`_ does not support multi-agent training we will use `RLlib <https://docs.ray.io/en/latest/rllib/index.html>`_ for this example. The following two methods run the same training. Running from the terminal may be more convenient for hyperparameter tuning, while running from the Unreal Editor may be more convenient when editing the game.

.. tabs::  
  
   .. group-tab:: Run from terminal  
  
      1. Run the game in Unreal Engine (by clicking the green triangle).  
      2. Open a terminal or command prompt, and run the following Python script:  
  
      .. code-block:: bash  
  
         schola-rllib -p 8000 -t 2000000 --use-attention

      3. Gradually increase the runner's speed in the ``Runner`` Blueprint → ``Character Movement: Walking`` → ``Max Walk Speed`` as the taggers improve and can consistently catch the slower runner.

      .. note::

         The ``--use-attention`` argument is used to enable the attention mechanism in RLlib. This gives temporal context to the agent allowing it to track the velocity of other agents, as well as not immediately forget prior observations, which can be crucial in complex environments. Its use is optional. Enabling it improves the agent's ability to navigate around obstacles, but will increase the number of training steps required.

   .. group-tab:: Run from Unreal Editor  

      Schola can also run the training script directly from the Unreal Editor. 
          
      #. Go to ``Edit`` → ``Project Settings``, and scroll down to find Schola.
      #. Check the :cpp:class:`Run Script on Play <UScholaManagerSubsystemSettings>` box.  
      #. Change :cpp:var:`~UScholaManagerSubsystemSettings::ScriptSettings` → :cpp:var:`~FScriptSettings::RLlibSettings` → :cpp:var:`~FRLlibTrainingSettings::Timesteps` to 2,000,000.
      #. Change :cpp:var:`~UScholaManagerSubsystemSettings::ScriptSettings` → :cpp:var:`~FScriptSettings::RLlibSettings` → :cpp:var:`~FRLlibNetworkArchSettings::bUseAttention` to true.
      #. Run the game in Unreal Engine (by clicking the green triangle).  
      #. Gradually increase the runner's speed in the ``Runner`` Blueprint → ``Character Movement: Walking`` → ``Max Walk Speed`` as the taggers improve and can consistently catch the slower runner is recommended.

      .. note::

         The :cpp:var:`~FRLlibNetworkArchSettings::bUseAttention` setting enables the attention mechanism in the RLlib. This gives temporal context to the agent allowing it to track the velocity of other agents, as well as not immediately forget prior observations, which can be crucial in complex environments. Its use is optional. Enabling it improves the agent's ability to navigate around obstacles, but will increase the number of training steps required.

      .. image:: /_static/guides/example_three/TagTrainingSettings.png   
      

Enabling TensorBoard
~~~~~~~~~~~~~~~~~~~~

To visualize the training progress, please refer to :ref:`Example 1 <maze_solver_tensorboard>` for details on using TensorBoard.
