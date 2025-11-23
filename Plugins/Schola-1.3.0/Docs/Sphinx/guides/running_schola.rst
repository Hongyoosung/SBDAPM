Running Schola
==============

Schola uses subclasses of :py:class:`~schola.core.unreal_connections.UnrealConnection` to interact with Unreal Engine. It supports both creating a standalone instance of Unreal Engine 
running as a child process of your python script, or connecting to a running Unreal Engine process.


Launch An Unreal Environment From Python
----------------------------------------

Schola supports running environments entirely from python using a :py:class:`~schola.core.unreal_connections.StandaloneUnrealConnection`.

.. code-block:: python

    from schola.core.unreal_connection import StandaloneConnection

    url="localhost" # Connect to the engine over localhost
    ue_path="Path to your Game Binary"
    port=None # Leave blank to use an arbitrary open port
    headless_mode = True # Should we skip rendering the Unreal Engine game?
    display_logs = False # Should we open a terminal to show Unreal Engine logs?
    _map=None # Run the default map for the selected executable, set to "/game/<path to map>" to use a different map.
    set_fps=60 # Set a fixed framerate for the game.
    disable_script=True # Ignore the RunScriptOnLaunch setting for the UnrealEngineGame.

    unreal_connection = StandaloneConnection(url=url,
                                            port=port,
                                            headless_mode=headless_mode,
                                            display_logs=display_logs,
                                            map=_map,
                                            set_fps=set_fps)


Initialize the Environment
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. tabs::

    .. group-tab:: Gymnasium
        
        .. code-block:: python

            from schola.gym.env import VecEnv
            from schola.core.unreal_connection import StandaloneConnection

            unreal_connection = StandaloneConnection("localhost","Path To Game Binaries", headless_mode=True)
            env = VecEnv(unreal_connection)
            ... # Your Code Here

    .. group-tab:: Ray
        
        .. code-block:: python

            from schola.ray.env import BaseEnv
            from schola.core.unreal_connection import StandaloneConnection

            unreal_connection = StandaloneConnection("localhost","Path To Game Binaries", headless_mode=True)
            env = BaseEnv(unreal_connection)
            ... # Your Code Here

    .. group-tab:: Stable Baselines 3

        .. code-block:: python

            from schola.sb3.env import VecEnv
            from schola.core.unreal_connection import StandaloneConnection

            unreal_connection = StandaloneConnection("localhost","Path To Game Binaries", headless_mode=True)
            env = VecEnv(unreal_connection)
            ... # Your Code Here


Connect To a Running Unreal Environment
---------------------------------------

Schola supports connecting to an already running Editor or Game, for debugging and Unreal Engine driven workflows using a :py:class:`~schola.core.unreal_connections.UnrealEditorConnection`


.. code-block:: python

    from schola.core.unreal_connection import UnrealEditorConnection

    url="localhost" # Connect to the engine over localhost
    port=8002 # Must match the port selected in your Unreal Engine Plugin Settings for Schola
    
    unreal_connection = UnrealEditorConnection(url, port)

Initialize the Environment
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. tabs::

    .. group-tab:: Gymnasium
        
        .. code-block:: python

            from schola.gym.env import VecEnv
            from schola.core.unreal_connection import UnrealEditorConnection

            unreal_connection = UnrealEditorConnection("localhost",8002)
            env = VecEnv(unreal_connection)
            ... # Your Code Here

    .. group-tab:: Ray
        
        .. code-block:: python

            from schola.ray.env import BaseEnv
            from schola.core.unreal_connection import UnrealEditorConnection

            unreal_connection = UnrealEditorConnection("localhost",8002)
            env = BaseEnv(unreal_connection)
            ... # Your Code Here

    .. group-tab:: Stable Baselines 3

        .. code-block:: python

            from schola.sb3.env import VecEnv
            from schola.core.unreal_connection import UnrealEditorConnection

            unreal_connection = UnrealEditorConnection("localhost",8002)
            env = VecEnv(unreal_connection)
            ... # Your Code Here