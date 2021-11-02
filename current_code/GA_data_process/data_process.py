# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 13:20:44 2021

@author: andub
"""

import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt('min4.txt')

plt.plot(data[1:, 0], data[1:, 4])