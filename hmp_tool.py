import os

import numpy as np

import config

print(config.ACTIONS)


def read_act_txt(act_dir):
    txt_dic = {}
    action_files = os.listdir(act_dir)
    for act_txt in action_files:
        act_txt_path = act_dir + "/" + act_txt
        np_arr = np.loadtxt(act_txt_path)
        # print(np_arr.shape)
        txt_dic.update({act_txt: np_arr})
    return txt_dic
