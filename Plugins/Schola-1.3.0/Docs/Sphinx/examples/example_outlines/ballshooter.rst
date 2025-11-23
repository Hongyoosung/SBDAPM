BallShooter
-----------

The BallShooter environment features a rotating turret that learns to aim and shoot at randomly moving targets. The agent can rotate in either direction, and detects the targets by using a cone shaped ray-cast.

To build the BallShooter environment from scratch, you can follow the guide available at :doc:`/examples/example_two`

.. csv-table::
    
    "Num Agents", "1"
    "Observation Space", "DictSpace({'Ray_Num_10_Deg_120,00_Max_4096,00_ECC_WorldStatic_Tags_Target': make_ray_cast_space(10,1,4096)})"
    "Action Space", "DictSpace({'BallShooter': DiscreteSpace(2),
    'DiscreteRotationActuator': DiscreteSpace(3)
    })"
    "Num Vectorized Copies", "Not Supported"


