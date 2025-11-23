MazeSolver
----------

The MazeSolver environment features a static maze that the agent learns to solve as fast as possible. The agent observers the environment using raycasts, moves by teleporting in 2 dimensions and is given a reward for getting closer to the goal.

To build the MazeSolver environment from scratch, you can follow the guide available at :doc:`/examples/example_one`

.. csv-table::
    :delim: |

    "Num Agents"| "1"
    "Observation Space"| "DictSpace({'Ray_Num_8_Deg_360,00_Max_4096,00_ECC_WorldStatic': make_ray_cast_space(num_rays=8,num_categories=0,max_dist=4096)})"
    "Action Space"| "DictSpace({'MovementInput\_XY\_\-10,00_10,00': BoxSpace([-10.0,-10.0],[10.0,10.0])})"
    "Num Vectorized Copies"| "16"


