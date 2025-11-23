Extending launch.py for Sb3 and Ray 
===================================

Schola supports extending the launch.py scripts for both ray and sb3 with additional callbacks and logging. This is done using `Python plugins <https://packaging.python.org/en/latest/guides/creating-and-discovering-plugins/>`_, which enable automatic discovery of new code extending schola. The plugins must extend an appropriate base class, and register an appropriately named entry point. The launch.py scripts will then automatically `discover the plugin <https://packaging.python.org/en/latest/specifications/entry-points/>`_ and use it to modify the training process. Below are the steps to specifically extend the ``schola.scripts.sb3.launch`` and ``schola.scripts.ray.launch`` scripts.

Extending schola.scripts.sb3.launch
-----------------------------------

You can extend ``schola.scripts.sb3.launch`` with additional callbacks, KVWriters for logging and command line arguments. Below is an example of how to implement a plugin which adds a CSV logger and a callback to log every N timesteps.
 
1. Create a new class that inherits from :py:class:`~schola.scripts.common.Sb3LauncherExtension`, and implement these methods if relevant: :py:meth:`~schola.scripts.common.Sb3LauncherExtension.get_extra_KVWriters`, :py:meth:`~schola.scripts.common.Sb3LauncherExtension.get_extra_callbacks`, and :py:meth:`~schola.scripts.common.Sb3LauncherExtension.add_plugin_args_to_parser`.

.. code-block:: python

    from schola.scripts.common import Sb3LauncherExtension
    from dataclasses import dataclass
    from typing import Dict, Any
    import argparse
    from stable_baselines3.common.logger import KVWriters, CSVOutputFormat
    from stable_baselines3.common.callbacks import LogEveryNTimesteps

    @dataclass
    class ExampleSb3Extension(Sb3LauncherExtension):
        csv_save_path: str = "./output.csv"
        log_frequency: int = 1000

        def get_extra_KVWriters(self):
            return [CSVOutputFormat(self.csv_save_path)]
        
        def get_extra_callbacks(self):
            return [LogEveryNTimesteps(n_steps=log_frequency)]

        @classmethod
        def add_plugin_args_to_parser(cls, parser: argparse.ArgumentParser):
            """
            Add example logging arguments to the parser.
            
            Parameters
            ----------
            parser : argparse.ArgumentParser
                The parser to which the arguments will be added.
            """
            group = parser.add_argument_group("CSV Logging")
            group.add_argument("--csv-save-path", type=str, help="The path to save the CSV file to")
            group.add_argument("--log-frequency", type=int, help="The frequency to log to the terminal")
            
2. Create a new Python package, with an entrypoint in the ``schola.plugins.sb3.launch`` group pointing to your new class.

.. code-block:: python

    setup(  
        ...,
        entry_points={  
            'schola.plugins.sb3.launch': [  
                'example_extension_name = example_plugin_name.example_extension_name:ExampleSb3Extension',  
            ],  
        },  
        ...,
    )  

Extending schola.scripts.ray.launch
-----------------------------------

You can extend ``schola.scripts.ray.launch`` with additional callbacks, and command line arguments. Below is an example of how to implement a plugin which adds support for logging with Wandb.

1. Create a new class that inherits from :py:class:`~schola.scripts.common.RLLibLauncherExtension`, and implement these methods if relevant: :py:meth:`~schola.scripts.common.RLLibLauncherExtension.get_extra_callbacks`, and :py:meth:`~schola.scripts.common.RLLibLauncherExtension.add_plugin_args_to_parser`.

.. code-block:: python

    from schola.scripts.common import RLLibLauncherExtension
    from dataclasses import dataclass
    from typing import Any, Dict, List
    import argparse
    from ray.tune.integration.wandb import WandbLoggerCallback  

    @dataclass
    class ExampleRayExtension(RLLibLauncherExtension):
        experiment_id: str = None

        def get_extra_callbacks(self):
            return [WandbLoggerCallback(project=self.experiment_id)]

        @classmethod
        def add_plugin_args_to_parser(cls, parser: argparse.ArgumentParser):
            """
            Add example logging arguments to the parser.
            
            Parameters
            ----------
            parser : argparse.ArgumentParser
                The parser to which the arguments will be added.
            """
            group = parser.add_argument_group("Wandb Logging")
            group.add_argument("--experiment-id", type=str, help="The experiment ID to log to")
            
2. Create a new Python package, with an entrypoint in the ``schola.plugins.ray.launch`` group pointing to your new class.

.. code-block:: python

    setup(  
        ...,
        entry_points={  
            'schola.plugins.ray.launch': [  
                'example_extension_name = example_plugin_name.example_extension_name:ExampleRayExtension',  
            ],  
        },  
        ...,
    )  

