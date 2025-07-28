import torch



MAPS = { # First index is a list that contains the indices of the observation, the second index is a list that contains the indices of the action 
    'a1': {
        'FL_hip': [[12, 24, 36], [0]],
        'FL_thigh': [[13, 25, 37], [1]],
        'FL_calf': [[14, 26, 38], [2]],
        'FR_hip': [[15, 27, 39], [3]],
        'FR_thigh': [[16, 28, 40], [4]],
        'FR_calf': [[17, 29, 41], [5]],
        'RL_hip': [[18, 30, 42], [6]],
        'RL_thigh': [[19, 31, 43], [7]],
        'RL_calf': [[20, 32, 44], [8]],
        'RR_hip': [[21, 33, 45], [9]],
        'RR_thigh': [[22, 34, 46], [10]],
        'RR_calf': [[23, 35, 47], [11]],
        'root': [list(range(0,12)), []],
    },
    'h1': {
        'left_hip_yaw_joint': [[12, 22, 32], [0]],
        'left_hip_roll_joint': [[13, 23, 33], [1]],
        'left_hip_pitch_joint': [[14, 24, 34], [2]],
        'left_knee_joint': [[15, 25, 35], [3]],
        'left_ankle_joint': [[16, 26, 36], [4]],
        'right_hip_yaw_joint': [[17, 27, 37], [5]],
        'right_hip_roll_joint': [[18, 28, 38], [6]],
        'right_hip_pitch_joint': [[19, 29, 39], [7]],
        'right_knee_joint': [[20, 30, 40], [8]],
        'right_ankle_joint': [[21, 31, 41], [9]],
        'root': [list(range(0,12)), []],
    },
    'g1': {
        'left_hip_pitch_joint': [[12, 24, 36], [0]],
        'left_hip_roll_joint': [[13, 25, 37], [1]],
        'left_hip_yaw_joint': [[14, 26, 38], [2]],
        'left_knee_joint': [[15, 27, 39], [3]],
        'left_ankle_pitch_joint': [[16, 28, 40], [4]],
        'left_ankle_roll_joint': [[17, 29, 41], [5]],
        'right_hip_pitch_joint': [[18, 30, 42], [6]],
        'right_hip_roll_joint': [[19, 31, 43], [7]],
        'right_hip_yaw_joint': [[20, 32, 44], [8]],
        'right_knee_joint': [[21, 33, 45], [9]],
        'right_ankle_pitch_joint': [[22, 34, 46], [10]],
        'right_ankle_roll_joint': [[23, 35, 47], [11]],
        'root': [list(range(0,12)), []],
    },
}


DIMS = {
    'a1': 48,
    'h1':42,
    'g1':48
}

def is_map_empty(map):
    for k, v in map.items():
        if len(v[0]) or len(v[1]):
            return False
    return True

class Mapping:

    def __init__(self, env_name, mask_empty=False):
        
        self.dim = DIMS[env_name]
        
        self.map = MAPS[env_name].copy()
        # self.shortest_path_matrix = SP_MATRICES[env_name]

        if mask_empty:
            mask_indices = []
            mask_keys = []
            for i, k, v in zip(range(len(self.map)), self.map.keys(), self.map.values()):
                if len(v[0]) == 0 and len(v[1]) == 0:
                    mask_indices.append(i)
                    mask_keys.append(k)
            
            # remove keys that are empty
            for key in mask_keys:
                self.map.pop(key)
                
            # keep_indices = [i for i in range(self.shortest_path_matrix.shape[0]) if i not in mask_indices]
            # self.shortest_path_matrix = self.shortest_path_matrix[keep_indices][:,keep_indices]

    def get_map(self):
        return self.map
    
    def create_observation(self, obs, stack_time=True):
        new_obs = {}
        for k, v in self.map.items():
            new_obs[k] = obs[:,v[0]]
            if stack_time:
                new_obs[k] = new_obs[k].reshape(obs.shape[0], -1)
        return new_obs
        