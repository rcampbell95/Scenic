import scenic
from scenic.simulators.mujoco.simulator import MujocoSimulator
from scenic.core.scenarios import Scene

if __name__ == "__main__":
    SAMPLES = 1

    for sample_index in range(SAMPLES):
        simulator = MujocoSimulator(xml="")
        scenario = scenic.scenarioFromFile("./examples/mujoco/simple_navigation.scenic")

        scene, _ = scenario.generate()
        simulation = simulator.simulate(scene, maxSteps=100000)

        result = simulation.result
