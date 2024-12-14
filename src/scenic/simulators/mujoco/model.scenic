import math
from collections.abc import Callable
from typing import List

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
            try:
                self.mjcf_model = mjcf.from_xml_string(xml)
            except ValueError as e:
                print(xml)
            self.body_name = self.mjcf_model.model + "/"
        else:
            self.mjcf_model = None    
    def model(self):
        return self.mjcf_model


class DynamicMujocoBody(MujocoBody):
    """Dynamic Mujoco Body"""
    def __init__(self, xml: str="", *args, **kwargs):
        super().__init__(xml, *args, **kwargs)

    def control(self, model, data):
        raise NotImplementedError("Error: control not implemented for object")
