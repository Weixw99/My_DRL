#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/5/22 0022 20:13
# @Author  : Chao Pan  
# @File    : markevery应用.py

import numpy as np
import matplotlib.pyplot as plt

# Create a figure of size 8x6 inches, 80 dots per inch
plt.figure(figsize=(8, 6), dpi=80)

# Create a new subplot from a grid of 1x1
plt.subplot(1, 1, 1)

# Set x limits
plt.xlim(-4.0, 4.0)
# Set x ticks
plt.xticks(np.linspace(-4, 4, 9, endpoint=True))

# set ylimits
plt.ylim(-1.2, 1.2)
# Set y ticks
plt.yticks(np.linspace(-1.2, 1.2, 5, endpoint=True))

X = np.linspace(-np.pi, np.pi, 256, endpoint=True)
C, S = np.cos(X), np.sin(X)

marker_on = [0, 5, 10, 15]

# Plot cosine with a blue continuous line of width 1 (pixels)
plt.plot(X, C, color="#fd3c06", linewidth=1.0, linestyle=':', marker='*', markevery=marker_on, label="cosine")
# plt.plot(X,C)

# Plot sine with a green continuous line of width 1 (pixels)
plt.plot(X, S, color="green", linewidth=1.0, linestyle="--", marker='s', markevery=marker_on, label="sine")
# Adding a legend
plt.legend(loc=5)

# Save figure using 100dots per inch

#plt.savefig("try.jpg", dpi=100, bbox_inches='tight')

#Show result on screen
plt.show()

