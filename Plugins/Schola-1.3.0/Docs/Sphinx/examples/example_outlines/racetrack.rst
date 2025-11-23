RaceTrack
---------

The RaceTrack environment features cars trained to follow a spline track. Cars can observe their absolute position as well as their velocity, and take action using inputs to a vehicle controller.

.. csv-table::
    
    "Num Agents", "1"
    "Observation Space", "DictSpace({'PositionObserver': BoxSpace(-100000.0, 100000.0, shape=(6,)),
    'VelocityObserver': BoxSpace(-100000.0, 100000.0, shape=(4,))
    })"
    "Action Space", "DictSpace({'VehicleController': BoxSpace(-1.0, 1.0, shape=(2,))})"
    "Num Vectorized Copies", "16"