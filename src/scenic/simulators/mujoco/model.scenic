import math

from scenic.core.object_types import Object
import dm_control 
from dm_control import mjcf
import numpy as np

import mujoco


class MujocoBody(Object):
    """Abstract class for Mujoco objects."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        xml = args[0]["xml"] if "xml" in args[0] else None
        self.mjcf_model = None
        self.elements = {}
        if xml:
            self.mjcf_model = mjcf.from_xml_string(xml)
            self.body_name = self.mjcf_model.model + "/"

            actuator = self.mjcf_model.actuator
        else:
            self.mjcf_model = None

        # Process model, keep list of bodies and actuators
    
    def model(self):
        return self.mjcf_model

    def control(self, model, data):
        return


class DynamicMujocoBody(MujocoBody):
    """Spheroid shape Mujoco Body"""
    def __init__(self, xml: str="", canApplyForce: bool=False, canApplyTorque: bool=False, canActuate: bool=False, *args, **kwargs):
        super().__init__(xml, *args, **kwargs)
        self.prev_torque = [0 for i in range(3)]
        self.prev_force = [0 for i in range(3)]
        self.prev_actuations = []

        self.canApplyForce = canApplyForce
        self.canApplyTorque = canApplyTorque
        self.canActuate = canActuate

    def applyForce(self, model, data):
        # Velocity of center of mass (COM) of body
        a, b, c = data.body(self.body_name).cvel[0:3]

        if data.time < 100:
            return [math.sin(data.time / 5), math.cos(data.time / 5), 0]

    def applyTorque(self, model, data):
        return [0, 0, 0]

    def control(self, model, data):
        force = [0 for i in range(3)]
        torque = [0 for i in range(3)]
        if self.canApplyForce:
            force = self.applyForce(model, data)
        if self.canApplyTorque:
            torque = self.applyTorque(model, data)

        xfrc_applied = force + torque
        data.body(self.body_name).xfrc_applied = xfrc_applied

        if self.canActuate and self.mjcf_model:
            for motor in self.mjcf_model.actuator.motor:
                actuator = data.actuator(f"{self.body_name}{motor.name}")
                actuator.ctrl = [np.random.random() - 0.5]

