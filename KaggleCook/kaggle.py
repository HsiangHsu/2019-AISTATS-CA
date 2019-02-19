#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 16:45:12 2018

@author: hsianghsu
"""

import tensorflow as tf
import numpy as np
from util import *
import json

with open('kaggle_cook_train.json', 'r') as outfile:
    data = json.load(outfile)

X_train = np.array(data['X'])
y_train = np.array(data['y'])

d = 10

session = tf.InteractiveSession()

(x_input,f_output,keepX) = SimpleNet(X_train.shape[1], name="Fnet", structure = [30, 30, 30, d])
(y_input,g_output,keepY) = SimpleNet(y_train.shape[1], name="Gnet", structure = [30, 30, 30, d])

lrate = tf.placeholder_with_default(1e-2,[])
objective, _ = create_loss_svd(f_output, g_output)
step = tf.train.GradientDescentOptimizer(lrate).minimize(objective)
session.run(tf.global_variables_initializer())


n_epochs = 20000
rate = 5e-3

for k in range(n_epochs):
    session.run(step,feed_dict= {x_input: X_train, y_input: y_train, lrate:rate})

f_train = session.run(f_output,feed_dict= {x_input: X_train})
g_train = session.run(g_output,feed_dict= {y_input: y_train})
F, G= whiten(f_train, g_train, f_train, g_train)

f_train_all = session.run(f_output,feed_dict= {x_input: np.identity(X_train.shape[1])})
g_train_all = session.run(g_output,feed_dict= {y_input: np.identity(y_train.shape[1])})
F_all, G_all = whiten(f_train_all, g_train_all, f_train, g_train)


data_out = {'F': F.tolist(), 'G': G.tolist(), 'F_all': F_all.tolist(), 'G_all': G_all.tolist()}
with open('kaggle_cook_FG'+str(n_epochs)+'.json', 'w') as outfile:
    json.dump(data_out, outfile)