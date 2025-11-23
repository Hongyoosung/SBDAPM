Basic
-----

The Basic environment features an agent that can move in the X-dimension and receives a small reward for going five steps in one direction and a bigger reward for going in the opposite direction.

.. csv-table::
    :delim: |

    "Num Agents"| "1"
    "Observation Space"| "DictSpace({'Position\_X\_\-500,00_500,00': BoxSpace(-500.0,500.0,)})"
    "Action Space"| "DictSpace({'Teleport Actuator': DiscreteSpace(3)}))"
    "Num Vectorized Copies"| "2"

