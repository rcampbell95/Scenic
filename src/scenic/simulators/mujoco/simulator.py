import math
import time
import copy

import mujoco
import mujoco.viewer
from dm_control import mjcf
from scipy.spatial.transform import Rotation
from scenic.core.type_support import toOrientation
from scenic.core.vectors import Vector
from scenic.core.simulators import Simulation, Simulator 


class MujocoSimulator(Simulator):
  def __init__(self, xml='', actual = False):
    super().__init__()
    self.xml=xml
    self.actual=actual
  
  def createSimulation(self, scene, **kwargs):
    return MujocoSimulation(scene, self.xml, self.actual, **kwargs)

class MujocoSimulation(Simulation):
  '''
  `Simulation` object for Mujoco.
  '''

  def __init__(self, scene, xml='', actual = False, **kwargs):
    self.xml=xml
    self.actual=actual
    self.scene=scene
    
    self.mujocohandle=None
    kwargs.pop('timestep')
    super().__init__(scene, timestep=.001, **kwargs)


  def setup(self):
    super().setup()
    if self.xml != "":
      self.model = mujoco.MjModel.from_xml_string(self.xml)
      self.data = mujoco.MjData(self.model)
    else:
      mjcf_model = mjcf.RootElement(model="model")

      mjcf_model.compiler.set_attributes(angle = "radian",
                                        coordinate = "local",
                                        inertiafromgeom = "true")

      mjcf_model.default.joint.set_attributes(armature = 0,
                                            damping = 1,
                                            limited = "false")

      mjcf_model.default.geom.set_attributes(friction = [0.5],
                                            solimp = [0.99, 0.99, 0.01],
                                            solref = [0.01, 0.5])

      mjcf_model.option.set_attributes(gravity=[0, 0, -9.8],
                                      timestep=self.timestep)

      mjcf_model.asset.add("texture",
                          type="skybox",
                          builtin="gradient",
                          rgb1=[1, 1, 1],
                          rgb2=[.6, .8, 1],
                          width=256,
                          height=256)

      mjcf_model.asset.add("texture",
                          name="texplane",
                          type="2d",
                          builtin="checker",
                          rgb1=[.2, .3, .4],
                          rgb2=[.1, .15, 2],
                          width=512,
                          height=512)

      mjcf_model.asset.add("material",
                          name="MatPlane",
                          texture="texplane",
                          texrepeat=[1, 1],
                          texuniform="true")

      mjcf_model.worldbody.add("light",
                              pos=[0, 1, 1],
                              dir=[0, -1, -1],
                              diffuse=[1, 1, 1])

      mjcf_model.worldbody.add("geom",
                              condim=3,
                              material="MatPlane",
                              name="ground",
                              pos=[0, 0, 0],
                              size=[1, 1, 0.1],
                              type="plane")


      for obj in self.objects:
        mjcf_model.worldbody.add("body",
                                name=f"{i}\_body",
                                pos=[obj.position[0], obj.position[1], obj.position[2]])

        mjcf_model.worldbody.body[i].add("joint",
                                        name=f"{i}\_joint",
                                        type="free",
                                        damping=0.001)

        quaternion = copy.copy(obj.orientation.q)
        shifted_quaternion = orientation.insert(0, orientation.pop())
        mjcf_model.worldbody.body[i].add("geom",
                                        name=f"{i}\_geom",
                                        quat=shifted_quaternion,
                                        size=[obj.width, obj.length, obj.height],
                                        rgba=obj.color,
                                        type="box",
                                        density=100)

      self.xml = mjcf_model.to_xml_string()

      self.model = mujoco.MjModel.from_xml_string(self.xml)
      self.data = mujoco.MjData(self.model)
   
   self.mujocohandle = mujoco.viewer.launch_passive(self.model, self.data)

  def createObjectInSimulator(self, obj):
    if self.mujocohandle == None: pass 
    else: 
      print("Mujoco does not handle creation of objects after intialization")
      return -1

  def step(self):
    mujoco.mj_step(self.model, self.data)
    self.mujocohandle.sync()
    if self.actual:
      time.sleep(self.timestep)
    
  def getProperties(self, obj, properties):
    j=1
    
    for checker in self.scene.objects:
      if checker == obj:
        # get postion
        x,y,z=self.data.geom(f'''{j}\_geom''').xpos
        position=Vector(x,y,z)

        # get angular velocity and speed
        a,b,c=self.data.qvel[3:6]
        angularVelocity = Vector(a,b,c)
        angularSpeed = math.hypot(*angularVelocity)

        # get velocity and speed
        a,b,c=self.data.qvel[0:3]
        velocity=Vector(a,b,c)
        speed = math.hypot(*velocity)

        cart_orientation=self.data.geom(f'''{j}\_geom''').xmat 
        a,b,c,d,e,f,g,h,i=cart_orientation
        new_mat = [[a,b,c],[d,e,f,],[g,h,i]]
        r = Rotation.from_matrix(new_mat)
        # ori = toOrientation(r)
        yaw, pitch, roll = obj.parentOrientation.localAnglesFor(r)
        break
      j=j+1
    values = dict(
      position=position,
      velocity=velocity,
      speed=speed,
      angularSpeed=angularSpeed,
      angularVelocity=angularVelocity,
      yaw=yaw,#x
      pitch=pitch, #x
      roll=roll,#x
    )
    return values


  def destroy(self):
    if self.mujocohandle != None:
      self.mujocohandle.close()
