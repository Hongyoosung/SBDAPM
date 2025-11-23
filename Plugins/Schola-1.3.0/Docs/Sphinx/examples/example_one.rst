Building Maze Solver
====================

This guide walks you through creating a maze environment and training an agent to navigate it using reinforcement learning in Unreal Engine.

.. image:: /_static/guides/example_one/maze_solver.gif  
  
In this example, we will create a static maze with a single agent that learns to navigate from the starting location (on the right side) to the goal location (on the left side) using reinforcement learning.  
The agent interacts with the environment by collecting observations through sensors and performing actions through actuators.  In this example, our agent will observe the walls around it using raycasts and move in the x and y directions.

We will train our agent by having it repeatedly try to solve the maze. Each attempt at the maze is referred to as an episode and ends when the agent successfully exits the maze, or runs out of time.

Periodically, the agent will review its performance during previous episodes and then update its policy to improve further. 
To quantify the performance of the agent we define a function that rewards the agent at each step of training. 
For this example, being away from the goal incurs a small penalty, hitting a wall incurs a medium penalty, and completing the maze results in a large reward. 
In this way the agent will learn a policy that maximizes the total reward received during each episode.
The agent can then use the learned policy during gameplay to decide which actions to take.

The Structure of the Environment in Unreal Engine
-------------------------------------------------
  
To build the game (called environment hereafter) where the agent will learn to solve the maze, we need the following in our Unreal Engine project:  
  
-  :localref:`Map<Creating the Map>`: The game map includes the floor and a maze constructed out of lots of walls. All objects such as the agent and environment definition will be placed in this map.  
-  :localref:`Agent blueprint<Creating the Agent>`: A subclass of `Character <https://dev.epicgames.com/documentation/en-us/unreal-engine/characters-in-unreal-engine>`_, which includes the shape, appearance, :cpp:class:`Sensors <USensor>`, and :cpp:class:`Actuators <UActuator>` of the agent.  

   -  :localref:`Sensor<Setting up Observation Collection>`: The agent has one :cpp:class:`Sensor <USensor>`, the :cpp:class:`RayCastObserver <URayCastObserver>`, which provides information about the agent's surroundings.
   -  :localref:`Actuator<Setting up Actuators>`: The agent has one :cpp:class:`Actuator <UActuator>`, the :cpp:class:`MovementInputActuator <UMovementInputActuator>`, which allows the agent to move in different directions.

-  :localref:`Trainer blueprint<Creating the Trainer>`: A subclass of :cpp:class:`BlueprintTrainer <ABlueprintTrainer>`, which includes the logic to compute the :cpp:func:`reward <ABlueprintTrainer::ComputeReward>` and :cpp:func:`status <ABlueprintTrainer::ComputeReward>` of the training.  
-  :localref:`Environment definition<Creating the Environment Definition>`: A subclass of :cpp:class:`BlueprintStaticScholaEnvironment <ABlueprintStaticScholaEnvironment>`, which includes the logic of :cpp:func:`initializing<ABlueprintStaticScholaEnvironment::InitializeEnvironment>` and :cpp:func:`resetting<ABlueprintStaticScholaEnvironment::ResetEnvironment>` the environment between different episodes of training.  
-  :localref:`Registering the agent<Registering the Agent>`: Connect the agent to the environment definition and trainer.  

.. _setup_unreal_engine:  

Initial Setup  
-------------  
  
1. Create a new blank project with a desired name and location.  
2. Install the Schola plugin to the project using the :doc:`/guides/setup_schola` guide.  
3. Go to ``Edit`` → ``Project Settings``, and scroll down to find Schola. 

   .. note::

      If you don't see Schola in the Project Settings, please check whether Schola is installed in ``Edit`` → ``Plugins Menu``. Please refer to the :doc:`/guides/setup_schola` guide for more information.

      .. image:: /_static/guides/example_one/plugin_menu.png
         :width: 450

4. For :cpp:var:`Gym Connector Class<UScholaManagerSubsystemSettings::GymConnectorClass>`, select :cpp:class:`Python Gym Connector <UPythonGymConnector>`  
  
.. image:: /_static/guides/example_one/create_blank_project.png  

.. image:: /_static/guides/example_one/schola_setting.png
  
Creating the Map  
----------------  
  
1. Create a wall blueprint class with collision enabled.  
2. Create a maze by arranging walls in the map scene.  
3. Optionally, add a finish line at the maze exit to visually mark the goal. 
4. Save the map as ``mazeMap``.  

.. image:: /_static/guides/example_one/maze_map.png    
   :width: 400  
  
Creating the Agent
------------------
  
1. Create a new Blueprint Class with parent class `Character <https://dev.epicgames.com/documentation/en-us/unreal-engine/characters-in-unreal-engine>`_, and name it ``MazeSolverAgent``.  
2. Add any desired `static mesh <https://dev.epicgames.com/documentation/en-us/unreal-engine/BlueprintAPI/StaticMesh>`_ as the agent’s body, and optionally select a good-looking material.  
3. Save and close the blueprint, and place a ``MazeSolverAgent`` at the starting location in the map.  
4. Check whether the location of the ``MazeSolverAgent`` has x=0. If not, move the entire maze with the agent to ensure the starting location has x=0.  
  
Setting up Observation Collection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:cpp:class:`Sensor <USensor>` objects are components that can be added to the agent or the :cpp:class:`BlueprintTrainer <ABlueprintTrainer>`. They can contain an :cpp:class:`Observer<UAbstractObserver>` object.
It informs the agent of the distances of the surrounding physical objects. 
The agent has one :cpp:class:`Sensor <USensor>`, containing the :cpp:class:`Ray Cast Observer <URayCastObserver>`. 
The observation from each ray includes whether the ray hits an object, and the distance of this object.  

.. note::

   :cpp:class:`Observer<UAbstractObserver>` objects can be attached in three ways:
   
   1. Attaching a :cpp:class:`Sensor <USensor>` component to the agent, which can contain an :cpp:class:`Observer<UAbstractObserver>` object.
   2. Attaching a :cpp:class:`Sensor <USensor>` component to the :cpp:class:`BlueprintTrainer <ABlueprintTrainer>`, which can contain an :cpp:class:`Observer<UAbstractObserver>` object.
   3. Adding directly in the :cpp:var:`~AAbstractTrainer::Observers` arrays in the :cpp:class:`BlueprintTrainer <ABlueprintTrainer>`.

.. image:: /_static/guides/example_one/maze_solver_sensor.png  
   :width: 300  
  
1. Open the ``MazeSolverAgent`` class in the blueprint editor.
2. Add a :cpp:class:`Sensor <USensor>` component.  
3. In ``Details`` → ``Sensor`` → ``Observer``, select :cpp:class:`Ray Cast Observer <URayCastObserver>`.  
4. In ``Details`` → ``Sensor`` → ``Observer`` → ``Sensor properties`` → :cpp:var:`~URayCastObserver::NumRays`, enter 8.  
5. In ``Details`` → ``Sensor`` → ``Observer`` → ``Sensor properties`` → :cpp:var:`~URayCastObserver::RayDegrees`, enter 360.  
6. In ``Details`` → ``Sensor`` → ``Observer`` → ``Sensor properties``, check the :cpp:var:`DrawDebugLines <URayCastObserver::bDrawDebugLines>` box.  
  
Setting up Actuators
~~~~~~~~~~~~~~~~~~~~
:cpp:class:`ActuatorComponent <UActuatorComponent>` can be added to the agent or the :cpp:class:`BlueprintTrainer <ABlueprintTrainer>`. They can contain an :cpp:class:`Actuator<UActuator>` object.
The agent has one :cpp:class:`Actuator <UActuator>`, the :cpp:class:`Movement Input Actuator <UMovementInputActuator>`. It allows the agent to move in different directions. In this tutorial, we will limit the agent to only move in the x and y direction.  

.. note::

   :cpp:class:`Actuator <UActuator>` objects can be attached in three ways:
   
   1. Attaching an :cpp:class:`ActuatorComponent <UActuatorComponent>` to the agent, which can contain an :cpp:class:`Actuator<UActuator>` object.
   2. Attaching an :cpp:class:`ActuatorComponent <UActuatorComponent>` component to the :cpp:class:`BlueprintTrainer <ABlueprintTrainer>`, which can contain an :cpp:class:`Actuator<UActuator>` object.
   3. Adding directly in the :cpp:var:`~AAbstractTrainer::Actuators` arrays in the :cpp:class:`BlueprintTrainer <ABlueprintTrainer>`.

.. image:: /_static/guides/example_one/maze_solver_actuator.png  
   :width: 300  
  
1. Open the ``MazeSolverAgent`` class in the blueprint editor.  
2. Add an :cpp:class:`Actuator <UActuator>` component.  
3. In ``Details`` → ``Actuator Component`` → ``Actuator``, select :cpp:class:`Movement Input Actuator <UMovementInputActuator>`.  
4. In ``Details`` → ``Actuator Component`` → ``Actuator`` → ``Actuator Settings``, uncheck :cpp:var:`HasZDimension <UMovementInputActuator::bHasZDimension>`.  
5. In ``Details`` → ``Actuator Component`` → ``Actuator`` → ``Actuator Settings``, set :cpp:var:`~UMovementInputActuator::Minspeed` to -10.  
6. In ``Details`` → ``Actuator Component`` → ``Actuator`` → ``Actuator Settings``, set :cpp:var:`~UMovementInputActuator::MaxSpeed` to 10.  
  
Creating the Trainer
--------------------

To train an agent in Schola, the agent must be controlled by an :cpp:class:`AbstractTrainer<AAbstractTrainer>`, which defines the :cpp:func:`~ABlueprintTrainer::ComputeReward` and :cpp:func:`~ABlueprintTrainer::ComputeStatus` functions.
In this tutorial, we will be creating an :cpp:class:`BlueprintTrainer <ABlueprintTrainer>` (subclass of :cpp:class:`AbstractTrainer<AAbstractTrainer>`).

1. Create a new Blueprint Class with parent class :cpp:class:`BlueprintTrainer <ABlueprintTrainer>`, and name it ``MazeSolverTrainer``.  
2. Add a new boolean variable. Name it ``hasHit``. This variable will store whether the agent has hit a wall in the current step. 
3. Set the Event Graph as shown below. This binds an `On Actor Hit <https://dev.epicgames.com/documentation/en-us/unreal-engine/BlueprintAPI/Collision/OnActorHit>`_ `event <https://dev.epicgames.com/documentation/en-us/unreal-engine/events-in-unreal-engine>`_ to our agent, allowing the reward function to detect when the agent hits a wall.

.. blueprint-file:: example_one/maze_solver_trainer_event_graph.bp
   :heading: MazeSolverTrainer > Event Graph
   :imagefallback:  /_static/guides/example_one/maze_solver_trainer.png
   :height: 400
   :zoom: -2


Define the Reward Function
~~~~~~~~~~~~~~~~~~~~~~~~~~

In this tutorial, we use a per-step reward for getting closer to the goalpost and one big reward for reaching the goalpost. Additionally, we give a penalty if the agent hits the wall. The per-step reward is computed as ``-abs(agentPositionX - goalpostPositionX) / envSize - hasHitWall`` if the agent has not reached the goalpost, and 10 if the agent has reached the goalpost.

1. Add a new float variable. Name it ``goalpostPositionX``. This variable will store the X-position of the goal post.
  
   1. Return to the map, and get the X-position of the end of the maze. 
   2. Return to the ``MazeSolverTrainer`` class, and set the default value of ``goalpostPositionX`` to this number.  

2. Add a new float variable. Name it ``envSize``. This variable will the width of the maze.

   1. Return to the map, and get the width the maze. 
   2. Return to the ``MazeSolverTrainer`` class, and set the default value of ``envSize`` to this number.  

3. Add a new local boolean variable in :cpp:func:`~ABlueprintTrainer::ComputeReward`. Name it ``CachedHasHit``. This is used to temporarily store the value of ``hasHit`` so we can reset it during :cpp:func:`~ABlueprintTrainer::ComputeReward`.
4. Set the :cpp:func:`~ABlueprintTrainer::ComputeReward` function as shown below.  
  
.. blueprint-file:: example_one/maze_solver_reward.bp
   :heading: MazeSolverTrainer > ComputeReward
   :imagefallback:  /_static/guides/example_one/maze_solver_reward.png
   :height: 400
   :zoom: -5

.. _maze_solver_status_function:

Define the Status Function  
~~~~~~~~~~~~~~~~~~~~~~~~~~  
  
There are three possible statuses for each time step:  
  
1. **Running**: The episode is still ongoing, and the agent continues interacting with the environment.  
2. **Completed**: The agent has successfully reached a terminal state, completing the episode.  
3. **Truncated**: The episode has been forcefully ended, often due to external limits like time steps or manual intervention, without reaching the terminal state.  
  
In this tutorial, the terminal state for the agent is reaching the maze exit, which we track by determining if the ``MazeSolverAgent`` has  ``X-position >= goalpostPositionX``
Thus, an episode is completed when the agent goes over the ``goalpostPositionX``. We also set a max step to prevent an episode from running indefinitely. 

1. Add a new integer variable. Name it ``maxStep``, and set the default value to 5000. This means an episode is truncated if it reaches 5000 time steps without completing. You may adjust this number if you want to allow longer or shorter episodes due to factors such as the size of the environment or the speed of the agent.
2. Set the :cpp:func:`~ABlueprintTrainer::ComputeStatus` as shown below.  
  

.. blueprint-file:: example_one/maze_solver_status.bp
   :heading: MazeSolverTrainer > ComputeStatus
   :imagefallback:  /_static/guides/example_one/maze_solver_status.png
   :height: 400
   :zoom: -5


.. note::

   The :cpp:var:`~AAbstractTrainer::Step` variable is a part of the :cpp:class:`BlueprintTrainer <ABlueprintTrainer>` and it tracks the current number of steps since the last :cpp:func:`~ABlueprintStaticScholaEnvironment::ResetEnvironment` call.


Creating the Environment Definition  
-----------------------------------  
  
To train an agent in Schola, the game must have an :cpp:class:`StaticScholaEnvironment<AStaticScholaEnvironment>` Unreal object, which contains the agent and logic for initializing or resetting the game environment. 
In this tutorial, we will be creating an :cpp:class:`Blueprint Environment<ABlueprintStaticScholaEnvironment>` (subclass of :cpp:class:`StaticScholaEnvironment<AStaticScholaEnvironment>`) as the Environment.
The :cpp:func:`~ABlueprintStaticScholaEnvironment::InitializeEnvironment` function is called at the start of the game, and sets the initial state of the environment.
In this tutorial, we save the initial location of the agent and `Set Global Time Dilation <https://dev.epicgames.com/documentation/en-us/unreal-engine/BlueprintAPI/Utilities/Time/SetGlobalTimeDilation>`_, which scales the time for all objects in the map to be 10x faster. This allows the agent to meaningfully explore more space during training, preventing the model from getting stuck at a local minimum, and decreasing training time.  
The :cpp:func:`~ABlueprintStaticScholaEnvironment::ResetEnvironment` function is called before every new episode. In this tutorial, we just reset the agent to its initial location.

1. Create a new Blueprint Class with parent class :cpp:class:`BlueprintStaticScholaEnvironment <ABlueprintStaticScholaEnvironment>`, and name it ``MazeSolverEnvironment``.  
2. Add a new variable named ``agentArray`` of type `Pawn (Object Reference) <https://dev.epicgames.com/documentation/en-us/unreal-engine/pawn-in-unreal-engine>`_ array. This variable keeps track of registered agents belonging to this environment definition.
  
   1. Make this variable publicly editable (by clicking on the eye icon to toggle the visibility).  
  
3. Add a new variable named ``agentInitialLocation`` of type `Transform <https://dev.epicgames.com/documentation/en-us/unreal-engine/BlueprintAPI/Math/Transform>`_. This variable is for storing the initial location of the agent, so it can be restored upon reset.  
4. Set the Event Graph and :cpp:func:`~ABlueprintStaticScholaEnvironment::RegisterAgents` function as shown below. 
5. Save and close the blueprint, and place a ``MazeSolverEnvironment``  anywhere in the map. The location does not matter.  
  

.. blueprint-file:: example_one/maze_solver_environment.bp
   :heading: MazeSolverEnvironment > Event Graph
   :imagefallback:  /_static/guides/example_one/maze_solver_environment.png
   :height: 400
   :zoom: -4

.. blueprint-file:: example_one/maze_solver_register_agent.bp 
   :heading: MazeSolverEnvironment > RegisterAgents
   :imagefallback:  /_static/guides/example_one/maze_solver_register_agent.png
   :height: 250
   :zoom: -1


Registering the Agent  
---------------------  
  
1. Click on the ``MazeSolverEnvironment``  in the map.  
  
   1. Go to ``Details panel`` → ``Default`` → ``Agent Array``.  
   2. Add a new element.  
   3. Select ``MazeSolverAgent`` in the drop-down menu.  
  
      .. image:: /_static/guides/example_one/maze_solver_environment_include_pawn.png  
         :width: 400  
  
2. Open the ``MazeSolverAgent`` class in the blueprint editor.  
  
   1. Go to Details Panel.  
   2. Search for `AIController <https://dev.epicgames.com/documentation/en-us/unreal-engine/ai-controllers-in-unreal-engine>`_.  
   3. In the drop-down, select ``MazeSolverTrainer`` .  
  
      .. image:: /_static/guides/example_one/maze_solver_aicontroller.png  
         :width: 400  
  
Starting Training   
-----------------  

We will train the agent using the `Proximal Policy Optimization (PPO) <https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html>`_ algorithm for 500,000 steps.
The following two methods run the same training. Running from the terminal may be more convenient for hyperparameter tuning, while running from the Unreal Editor may be more convenient when editing the game.

.. tabs::  
  
   .. group-tab:: Run from terminal  
  
      1. Run the game in Unreal Engine (by clicking the green triangle).  
      2. Open a terminal or command prompt, and run the following Python script:  
  
      .. code-block:: bash  
  
         schola-sb3 -p 8000 -t 500000 PPO  

   .. group-tab:: Run from Unreal Editor  

      Schola can also run the training script directly from the Unreal Editor. 
          
      1. Go to ``Edit`` → ``Project Settings``, and scroll down to find Schola.
      2. Check the :cpp:class:`Run Script on Play <UScholaManagerSubsystemSettings>` box.  
      3. Change :cpp:var:`~UScholaManagerSubsystemSettings::ScriptSettings` → :cpp:var:`~FScriptSettings::SB3Settings` → :cpp:var:`~FSB3TrainingSettings::Timesteps` to 500000.
      4. Run the game in Unreal Engine (by clicking the green triangle).  

      .. note::
   
         By default, Schola runs the ``python`` command when launching Python. If you have Python installed differently, such as ``python3.9``, or ``/usr/bin/python3``, 
         please change

         1. :cpp:var:`~UScholaManagerSubsystemSettings::ScriptSettings` → :cpp:var:`~FScriptSettings::EnvType` to ``Custom Python Path``.
         2. :cpp:var:`~UScholaManagerSubsystemSettings::ScriptSettings` → :cpp:var:`~FScriptSettings::CustomPythonPath` to your Python path or alias.
         

      .. image:: /_static/guides/example_one/running_from_editor.png   
        
.. _maze_solver_tensorboard:

Enabling TensorBoard
~~~~~~~~~~~~~~~~~~~~

TensorBoard is a visualization tool provided by TensorFlow that allows you to track and visualize metrics such as loss and reward during training. 

.. tabs::  
  
   .. group-tab:: Run from terminal  

      Add the ``--enable-tensorboard`` flag to the command to enable TensorBoard. The ``--log-dir`` flag sets the directory where the logs are saved.

      .. code-block:: bash  
  
         schola-sb3 -p 8000 -t 500000 --enable-tensorboard --log-dir experiment_maze_solver PPO  

   .. group-tab:: Run from Unreal Editor  

      Schola can also run the training script directly from the Unreal Editor. 
          
      1. Go to ``Edit`` → ``Project Settings``, and scroll down to find Schola.
      2. Check the :cpp:var:`~UScholaManagerSubsystemSettings::ScriptSettings` → :cpp:var:`~FScriptSettings::SB3Settings` → :cpp:var:`~FSB3TrainingSettings::LoggingSettings` → :cpp:var:`SaveTBLogs<FSB3LoggingSettings::bSaveTBLogs>` box.
      3. Set the :cpp:var:`~UScholaManagerSubsystemSettings::ScriptSettings` → :cpp:var:`~FScriptSettings::SB3Settings` → :cpp:var:`~FSB3TrainingSettings::LoggingSettings` → :cpp:var:`SaveTBLogs<FSB3LoggingSettings::LogDir>` to ``experiment_maze_solver`` or another location for the logs to be saved.
      4. Run the game in Unreal Engine (by clicking the green triangle).           

      .. image:: /_static/guides/example_one/running_from_editor_with_tensorboard.png   

After training, you can view the training progress in TensorBoard by running the following command in the terminal or command prompt. Make sure to first `install TensorBoard <https://pypi.org/project/tensorboard>`_, and set the ``--logdir`` to the directory where the logs are saved. 

.. code-block:: bash

   tensorboard --logdir experiment_maze_solver/PPO_1

.. note::

   Logs for subsequent runs will be in ``PPO_2``, ``PPO_3``, etc.

.. image:: /_static/guides/example_one/maze_solver_tensorbard.png  


Next Steps
----------

Congratulations! You have trained your first Schola agent!  From here, you can try the following:

1. Modify the :cpp:func:`reward <ABlueprintTrainer::ComputeReward>` to be only sparse rewards, and see how the agent performs after retrain.
2. Add more sensors to the agent or modify the :cpp:class:`RayCastObserver <URayCastObserver>` parameters, and see how the agent performs after retrain.
3. Change the initial location of the agent for every episode, and see how the agent performs after retrain.
4. Advanced: dynamically change the maze shape (same size or different sizes) for every episode, and try to train the agent to solve all kinds of mazes.