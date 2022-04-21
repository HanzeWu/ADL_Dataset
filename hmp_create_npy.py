import config
import numpy as np
from hmp_tool import read_act_txt

seg_root_path = config.HMP_segment_config
segment_len = config.TIME_SEQ_LENGTH
action = "Drink_glass"
action_file = action + ".txt"
act_array = np.loadtxt(seg_root_path + action_file, dtype=np.int)
action_dir = config.HMP + action
txts_dict = read_act_txt(action_dir)
action_np_list = []
for idx, txt_dict in enumerate(txts_dict.items(), 0):
    act_txt, np_arr = txt_dict[0], txt_dict[1]
    # print(np_arr.shape)
    start_idx = act_array[idx]
    print(start_idx)
    end_idx = start_idx + config.TIME_SEQ_LENGTH
    # print(type(np_arr[start_idx:start_idx + config.TIME_SEQ_LENGTH, :]))
    act_np = np.array(np_arr[start_idx:end_idx, :])
    if act_np.shape[0] < config.TIME_SEQ_LENGTH:
        print(act_txt)
        print("please change the start idx in line [%d] of %s !!!!" % (idx + 1, action_file))
    print(act_np.shape)
    action_np_list.append(act_np)
action_np = np.stack(action_np_list)
print(action_np.shape)
print(type(action_np))
np.save(config.HMP_npy_my + "/" + action + ".npy", action_np)
print(action + " npy saved over!")
