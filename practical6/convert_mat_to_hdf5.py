#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import scipy.io
import h5py

mat = scipy.io.loadmat("set1_1.mat")
motionArray = mat["X"]
roundedEventArray = mat["y"]
nSamples, nFeatures, nSubjects = motionArray.shape

h5f = h5py.File("set1_1.h5", "w")
for subject in range(nSubjects):
    X = motionArray[:,:,subject]
    y = roundedEventArray[:,:,subject]

    group = str(subject + 1) + "/"
    h5f.create_dataset(group + "X", data=X)
    h5f.create_dataset(group + "y", data=y)
