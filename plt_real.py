import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

import config
from hmp_tool import read_act_txt

print(config.ACTIONS)


def plt_real(act_txt, np_arr):
    x = np_arr[:, 0]
    y = np_arr[:, 1]
    z = np_arr[:, 2]
    plt.Figure()
    plt.plot(x, linewidth=1, color='red', )
    plt.plot(y, linewidth=1, color='green')
    plt.plot(z, linewidth=1, color='blue')
    x_major_locator = MultipleLocator(50)
    # Set the x scale interval to 1 and store it in the variable
    y_major_locator = MultipleLocator(10)
    # Set the y scale interval to 10 and store it in the variable
    ax = plt.gca()
    # ax is an instance of the two axes
    ax.xaxis.set_major_locator(x_major_locator)
    # Set the primary scale on the X-axis to a multiple of 1
    ax.yaxis.set_major_locator(y_major_locator)
    # Set the primary scale on the Y-axis to a multiple of 10
    plt.savefig(act_txt)
    print("saved in ", act_txt)
    # plt.show()
    # clear
    plt.clf()


for action in config.ACTIONS:
    action_dir = config.HMP + action
    txt_dict = read_act_txt(action_dir)
    for act_txt, np_arr in txt_dict.items():
        plt_real(config.HMP_real + action + "/" + act_txt + ".png", np_arr)
print("plt_real is over!")
