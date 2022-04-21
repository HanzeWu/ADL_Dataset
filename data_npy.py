import numpy as np
import config
from hmp_tool import read_act_txt

for action in config.ACTIONS:
    action_dir = config.HMP + action
    txt_dict = read_act_txt(action_dir)
    action_np_list = []
    for act_txt, np_arr in txt_dict.items():
        # print(np_arr.shape)
        start_idx = config.ACTIONS_FEATURE_RANGE[action]
        end_idx = start_idx + config.TIME_SEQ_LENGTH
        # print(type(np_arr[start_idx:start_idx + config.TIME_SEQ_LENGTH, :]))
        act_np = np.array(np_arr[0:150, :])
        print(act_np.shape)
        action_np_list.append(act_np)
    action_np = np.stack(action_np_list)
    print(action_np.shape)
    print(type(action_np))
    np.save(config.HMP_npy + "/" + action + ".npy", action_np)
    print(action + " npy saved over!")

# n = np.load(config.HMP_npy + "/Walk.npy")
