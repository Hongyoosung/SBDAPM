3DBall
------

The 3DBall environment features an agent that is trying to balance a ball on-top of itself. The agent can rotate itself and receives a reward every step until the ball falls.

.. csv-table::
    :delim: |
    
    "Num Agents"| "1"
    "Observation Space"| "DictSpace({'Position\_X\_\-500,00_500,00\_Y\_\-500,00_500,00\_Z\_\-500,00_500,00_Other_Relative': BoxSpace(-500.0,500.0,shape=(3,)),
    'Rotation\_Pitch\_\-180,00_180,00\_Yaw\_\-180,00_180,00\_Roll\_\-180,00_180,00': BoxSpace(-180.0,180.0,shape=(3,)),
    'Velocity\_X\_\-20,00_20,00\_Y\_\-20,00_20,00\_Z\_\-20,00_20,00_Other': BoxSpace(-20.0,20.0,shape=(3,))
    })"
    "Action Space"| "DictSpace({'Rotation Actuator': BoxSpace(-10.0,10.0,shape=(2,))})"
    "Num Vectorized Copies"| "3"

