Tag
---

The Tag environment features a 3v1 game of tag, where one agent(the runner) has to run away from the other agents which are trying to collide with it. The agents move using forward, left and right movement input, and observe the environment with a combination of ray-casts and global position data.

To build the Tag environment from scratch, you can follow the guide available at :doc:`/examples/example_three`

.. csv-table::
    :delim: |

    "Num Agents"| "4"
    "Observation Space (Tagger)"| "DictSpace({
    'Ray_Num_36_Deg_360,00_Max_2048,00_ECC_WorldStatic_Tags_Runner_Tagger': make_ray_cast_space(num_rays=36,num_categories=2,max_dist=2048),
    'RunnerSensor': BoxSpace(-50000.0,50000.0,shape=(4,)),
    'TeammateSensor 1': BoxSpace(-50000.0,50000.0,shape=(4,)),
    'TeammateSensor 2': BoxSpace(-50000.0,50000.0,shape=(4,))
    })"
    "Observation Space (Runner)"| "DictSpace({'Ray_Num_36_Deg_360,00_Max_2048,00_ECC_WorldStatic_Tags_Runner_Tagger': make_ray_cast_space(36,2,2048)})"
    "Action Space"| "DictSpace({'MovementInput\_X\_0,00_1,00': BoxSpace(0.0,1.0), 'MovementInput\_Y\_\-1,00_1,00': BoxSpace(-1.0,1.0)})"
    "Num Vectorized Copies"| "2"
