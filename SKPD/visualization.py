from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import time
import os
# from criterion import *
# from my_operator import *
import seaborn as sns
import k3d

def vis_tensor(beta):
    volume = k3d.volume(beta)
    plot = k3d.plot(camera_auto_fit=True)
    plot += volume
    plot.lighting = 2
    plot.display()