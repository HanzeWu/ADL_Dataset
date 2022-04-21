import os

###########################
## set path
###########################
DATA_ROOT = r'./data/'
HMP = DATA_ROOT + 'HMP/'
HMP_npy = DATA_ROOT + 'HMP_npy/'
HMP_npy_my = DATA_ROOT + 'HMP_npy_my/'
HMP_real = DATA_ROOT + 'HMP_real/'
HMP_RP = DATA_ROOT + 'HMP_RP/'
HMP_segment_config = DATA_ROOT + 'HMP_segment_config/'
ACTIONS = os.listdir(HMP)


def init_dir():
    if not os.path.exists(HMP_npy):
        os.mkdir(HMP_npy)
    if not os.path.exists(HMP_real):
        os.mkdir(HMP_real)
    if not os.path.exists(HMP_RP):
        os.mkdir(HMP_RP)
    if not os.path.exists(HMP_segment_config):
        os.mkdir(HMP_segment_config)
    if not os.path.exists(HMP_npy_my):
        os.mkdir(HMP_npy_my)
    for ACTION in ACTIONS:
        real = HMP_real + ACTION
        RP = HMP_RP + ACTION
        if not os.path.exists(real):
            os.mkdir(real)
        if not os.path.exists(RP):
            os.mkdir(RP)
    # print("data dir is ok!")

init_dir()

###########################
## set data feature
###########################
TIME_SEQ_LENGTH = 150
START_IDX = 0
##########################
## model params
#########################
image_size = (224, 224)
batch_size = 8
test_size = 0.3
