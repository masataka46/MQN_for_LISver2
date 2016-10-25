

# MQN for LIS ver2

This software is a python implementation of MQN for LIS ver2.

The original program is LIS ver2 or LIS.
https://github.com/wbap/lis

I modified his program to include MQN unit.
MQN is presented by J. Oh, et al in their literature 'Control of Memory, Active Perception, and Action in Minecraft'.
https://arxiv.org/abs/1605.09128

Requirement and 'how to run' is equivalent to LIS ver2.

Before you excute this code, you have to copy these files to gym_client/examples/agents/ root.

In this code, the time span of memory unit is set to 10. For example, if you want to change this to 20, the constant 'time_span' in several files should be 20, and the constant time_M should be 21.
