Setting Up Inference
====================

This guide will explain how to use your trained RL agents in inference mode (i.e. without connecting to Python).


.. note:: 
    
    This guide assumes you have already done a training run using Schola and have either a saved checkpoint have already exported the model to Onnx.


Convert a Checkpoint to Onnx
----------------------------

If you did not export to onnx during training you will need to convert a checkpoint to Onnx, you can use the following scripts to create an Onnx model from your checkpoint:

.. tabs::

    .. group-tab:: Stable Baselines 3
        .. code-block:: bash
            
            python -m ./Resources/python/scripts/sb3/sb3-to-onnx.py --model-path <CHECKPOINT_PATH> --output-path <ONNX_PATH>
    
    .. group-tab:: Ray
        .. code-block:: bash

            python -m ./Resources/python/scripts/sb3/rllib-to-onnx.py --policy-checkpoint-path <CHECKPOINT_PATH> --output-path <ONNX_PATH>

These commands will create an Onnx model in a standardized format compatible with Schola that can be used in the next section.

Load an Onnx Model into Unreal Engine
-------------------------------------

Once you have your Onnx model you can import it into Unreal Engine by dragging and dropping the `.onnx` file into the content browser. This will create a new Onnx model data asset in your project.


Creating an Inference Agent
---------------------------

In Schola an inference agent is any object implementing the :cpp:class:`Agent Interface <IInferenceAgent>`. However, we also provide three default implementations to help get the user started :cpp:class:`InferenceComponent <UInferenceComponent>`, :cpp:class:`InferencePawn <AInferencePawn>`, and :cpp:class:`InferenceController <AInferenceController>`. Follow the instructions below to prepare an actor or pawn to be controlled by your trained policy.

.. tabs::

    .. group-tab:: Inference Component
        
        To use the Inference Component, add a :cpp:class:`UInferenceComponent` to any actor you want to control with your trained policy.

    .. group-tab:: Inference Pawn
        
        To use the Inference Pawn, create a new Pawn inheriting from :cpp:class:`AInferencePawn`. This will create a pawn that can be controlled by your trained policy.

    .. group-tab:: Inference Controller

        To use the Inference Controller, create a child of :cpp:class:`AInferenceController` and set the controller of the pawn you want to control to your new class.


Once you've created your Inference Agent, go to the `Reinforcement Learning` tab of the details panel and set the Onnx model Property under Policy to the Data Asset created when you imported your onnx model into unreal.