import unittest
import numpy as np
import flygym
from flygym.envs.nmf_mujoco import NeuroMechFlyMuJoCo


random_state = np.random.RandomState(0)


class MuJoCoTest(unittest.TestCase):
    def test_untethered_sinewave(self):
        nmf = NeuroMechFlyMuJoCo()
        run_time = 0.01
        freq = 500
        phase = 2 * np.pi * random_state.rand(len(nmf.actuators))
        amp = 0.9
        
        while nmf.curr_time <= run_time:
            joint_pos = amp * np.sin(freq * nmf.curr_time + phase)
            action = {'joints': joint_pos}
            obs, info = nmf.step(action)
            nmf.render()
        nmf.close()


if __name__ == '__main__':
    unittest.main()