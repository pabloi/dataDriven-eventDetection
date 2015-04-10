#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import scipy.io
import h5py

mat = scipy.io.loadmat("dataArrays.mat")
motionArray = mat["motionArray"]
roundedEventArray = mat["roundedEventArray"]

nSubjects = 8
nTrials = 3

h5f = h5py.File("data.h5", "w")
for subject in range(nSubjects):
    for trial in range(nTrials):
        X = motionArray[:-1,:,subject,trial]
        y = roundedEventArray[:-1,:,subject,trial]

        group = str(subject + 1) + "/" + str(trial + 1) + "/"
        h5f.create_dataset(group + "X", data=X)
        h5f.create_dataset(group + "y", data=y)
