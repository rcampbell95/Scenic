import scenic
from scenic.simulators.mujoco.simulator import MujocoSimulator
from scenic.core.scenarios import Scene

if __name__ == "__main__":
    SAMPLES = 5

    for sample_index in range(SAMPLES):
        simulator = MujocoSimulator(xml="", actual=False, use_default_arena=False)
        scenario = scenic.scenarioFromFile("./examples/mujoco/inverted_pendulum.scenic")

        scene, _ = scenario.generate()
        simulation = simulator.simulate(scene, maxSteps=100000)

        result = simulation.result
