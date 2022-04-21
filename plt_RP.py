import matplotlib.pyplot as plt
import numpy as np
from pyts.image import RecurrencePlot
import config
#reference for pyts package
#@article{JMLR:v21:19-763,
# author  = {Johann Faouzi and Hicham Janati},
# title   = {pyts: A Python Package for Time Series Classification},
#  journal = {Journal of Machine Learning Research},
# year    = {2020},
# volume  = {21},
# number  = {46},
# pages   = {1-6},
# url     = {http://jmlr.org/papers/v21/19-763.html}
#}

def plt_rp(action):

    # data = np.load(config.HMP_npy + "./" + action + ".npy")
    data = np.load(config.HMP_npy_my + "./" + action + ".npy")
    # Recurrence plot transformation
    # time_delay=1 to img_size = 150 - time_delay
    rp = RecurrencePlot(dimension=2, percentage=10,time_delay=1)
    X = data[:, :, 0]
    Y = data[:, :, 1]
    Z = data[:, :, 2]
    X_rp = rp.fit_transform(X)
    Y_rp = rp.fit_transform(Y)
    Z_rp = rp.fit_transform(Z)

    imgs = np.stack([X_rp, Y_rp, Z_rp]).transpose(1, 2, 3, 0)  # sample_num,img_size,img_size,channel
    imgs = (imgs / (np.max(imgs) - np.min(imgs))) * 255 # normalization
    imgs = imgs.astype(int)
    # print("max:", max(imgs))
    # print("min:", min(imgs))

    for idx in range(len(imgs)):
        plt.close('all')
        plt.axis('off')
        plt.margins(0, 0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.imshow(imgs[idx])
        plt.savefig(config.HMP_RP + action + '/{}{}.png'.format(action, idx), bbox_inches='tight', pad_inches=0)
        plt.close('all')


# Show the results for the first time series
# plt.close('all')
# plt.axis('off')
# plt.margins(0, 0)
# plt.gca().xaxis.set_major_locator(plt.NullLocator())
# plt.gca().yaxis.set_major_locator(plt.NullLocator())
# plt.imshow(imgs[0])
# plt.close('all')

for action in config.ACTIONS:
    plt_rp(action)
    print(action, "rp over!")
