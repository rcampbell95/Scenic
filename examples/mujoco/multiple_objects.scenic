from scenic.simulators.mujoco.model import *


ego = new MujocoBody at (0, 0, 5),
        with width 0.5,
        with length 0.5,
        with height 0.5,
        with color (0.5, 0.5, 0.5, 1),
        with shape SpheroidShape(),
        with canApplyForce True,
        with canApplyTorque True
        #facing (-90 deg, 45 deg, 0)


obstacle1 = new MujocoBody at (-2, 0, 3),
    with color (0.75, 0.5, 0.5, 1),
    with width 1,
    with length 1,
    with height 1,
    with shape BoxShape()


new MujocoBody at (0, 2, 0),
        with width 0.5,
        with length 0.5,
        with height 0.5,
        with color (0.5, 0.5, 0.5, 1),
        with shape SpheroidShape(),
        #facing (-90 deg, 45 deg, 0)

new MujocoBody at (0, 4, 0),
        with width 0.5,
        with length 0.5,
        with height 0.5,
        with color (0.5, 0.5, 0.5, 1),
        with shape SpheroidShape(),

        #facing (-90 deg, 45 deg, 0)

