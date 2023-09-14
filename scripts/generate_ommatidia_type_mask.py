import numpy as np
from flygym.util.config import num_ommatidia_per_eye
from flygym.util.data import data_path


pale_yellow_ratio = [0.3, 0.7]
num_pale = int(num_ommatidia_per_eye * pale_yellow_ratio[0] / sum(pale_yellow_ratio))
num_yellow = num_ommatidia_per_eye - num_pale
pale_mask = np.concatenate([np.ones(num_pale), np.zeros(num_yellow)]).astype(bool)
np.random.seed(0)
np.random.shuffle(pale_mask)
np.save(data_path / "vision/pale_mask.npy", pale_mask)
