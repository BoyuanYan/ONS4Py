import torch
import torch.nn as nn
import subprocess as sp
import matplotlib.pyplot as plt
import os


def parse_log(file):
    """
    解析log文件，画出阻塞率的变化曲线
    :param file:
    :return:
    """
    prefix = 'bash'
    log_file = os.path.join(prefix, file)
    out = sp.getoutput("cat {}| grep remain".format(log_file))
    out = out.split('\n')
    y = []
    for i in out:
        tmp = i.split(' ')[26]
        tmp = tmp.split('=')[1]
        y.append(float(tmp))
    plt.plot(y)
