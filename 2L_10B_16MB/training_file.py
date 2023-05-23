# -*- coding: utf-8 -*-
"""COD_SpeechFile.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1X9TWM1_HmD00We4l2kiVlVaX7b0an-sF
"""

import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
"""
import random
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as data
import math
import pickle
from IPython.display import Audio

# -----------------------
# Parameter definition
# -----------------------

seed = 3

if seed != None:
    model_name = 'model_2L_10B_16MB_R'+str(seed)+'.dat'
else:
    model_name = 'model_2L_10B_16MB.dat'

input_dim = 1
output_dim = 256
hidden_dim = 256
nb_lstm_layers = 2
nb_epochs = 10
batch_size = 16

print('  input dimension: %d' % input_dim)
print('  hidden dimension: %d' % hidden_dim)
print('  output dimension: %d' % output_dim)
print('  number of LSTM layers: %d' % nb_lstm_layers)
print('  number of epochs: %d' % nb_epochs)
print('  batch size: %d' % batch_size)

# -----------------------------------------------------------------------------
# Data -> Artificial data
# This is sequence regression example, input is continous and output is continous.

# !wget http://dihana.cps.unizar.es/~cadrete/speech.pkl --no-check-certificate
# -----------------------------------------------------------------------------


with open('/home/alumnos/alumno3/work/TFM/data/speech.pkl','rb') as f:
    x = pickle.load(f)

#print(x.shape)

T = len(x[0])
N = len(x)
#print("N muestras/señal: ", T)
#print("N señales: ", N)

def dec(x, M=256):
    mu = M - 1
    y = x.astype(np.float32)
    y = 2 * (y / mu) - 1
    x = np.sign(y) * (1.0 / mu) * ((1.0 + mu)**abs(y) - 1.0)
    return x

x[:,0] = 0

#print(x.min(), x.max())

fs = 8000
x_manipulate = dec(x)



#plt.figure()
#plt.plot(x_manipulate[0,:])
#plt.savefig('audiosignal.png')
#Audio(x_manipulate[0,:],rate=fs)

# --------------------------
# Train
# Train, Dev split
# --------------------------

x = x.reshape(N, T, input_dim)


if seed != None:
    random.seed(seed)
    random.shuffle(x)
    print("Random seed = ", seed)

x_test = x[:100]
x_dev = x[100:200]
x_train = x[200:]
#x_dev_aux = x_dev[:10,:100]
#x_train_aux = x_train[:10,:100]


print('  x_test: %s (%s)' % (x_test.shape, x_test.dtype))
print('  x_dev: %s (%s)' % (x_dev.shape, x_dev.dtype))
print('  x_train: %s (%s)' % (x_train.shape, x_train.dtype))
#print('  x_dev_aux: %s (%s)' % (x_dev_aux.shape, x_dev_aux.dtype))
#print('  x_train_aux: %s (%s)' % (x_train_aux.shape, x_train_aux.dtype))
"""

# -------------
# Net
# -------------

class Net(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, nb_lstm_layers, dropout=0.5):
        super(Net, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, nb_lstm_layers, batch_first=True, bidirectional=False, dropout=dropout)
        self.fc = nn.Linear( hidden_dim, output_dim)
        self.J = nn.CrossEntropyLoss()

    def forward(self, x):
        x = x.float()
        x, state = self.lstm(x)
        x = self.fc(x)
        return x

    def loss(self, out, y): 
        n, t, d = y.shape
        return self.J(out.reshape(n*t,-1), y.reshape(n*t,))

    def predict(self, x, state=None):
        x = x.float()
        x, state = self.lstm(x, state)
        x = self.fc(x)
        return x.softmax(-1), state
"""

model = Net(input_dim, hidden_dim, output_dim, nb_lstm_layers)
model.cuda()
nb_param = sum(p.numel() for p in model.parameters())
print(model)
print('# param:    %d' % nb_param)


# -----------------------------------------------------
# Optimizer
# Scheduler decreases lr every epoch by a factor of 0.1
# -----------------------------------------------------

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
#optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)


# -----------------
# Train
# -----------------


trainloader = data.DataLoader(x_train, batch_size=batch_size, shuffle=True, num_workers=0)
devloader = data.DataLoader(x_dev, batch_size=1, shuffle=False, num_workers=0)

loss_train = np.zeros(nb_epochs)
loss_dev = np.zeros(nb_epochs)

model.train()
for i in range(nb_epochs):
    if i > 0:
        scheduler.step()

    # ---- W
    tic = time.time()
    model.train()
    i=1
    for x in trainloader:
        x = x.cuda()
        y = x[:,1:]      # value to predict
        x = x[:,:-1]     # previous values

        optimizer.zero_grad()
        out = model(x)
        #print(out.shape)
        loss = model.loss(out, y.long())
        loss.backward()
        optimizer.step()

        loss_train[i] += loss.item() /  (len(x_dev) * x.shape[0] * x.shape[1])
    toc = time.time()

    # ---- print
    #print("it %d/%d, Jtr = %f, time: %.2fs" % (i, nb_epochs, loss_train[i], toc - tic))

    # ---- dev
    model.eval()
    for x in devloader:
        x = x.cuda()

        y = x[:,1:]      # value to predict
        x = x[:,:-1]     # previous values

        out = model(x)
        loss = model.loss(out, y.long())

        loss_dev[i] += loss.item() / (len(x_dev) * x.shape[0] * x.shape[1])

        #print('    Jdev = %f, err = %f\n' % (loss_dev[i], err_dev[i]))
    #print('    Jdev = %f\n' % loss_dev[i])


# ------------------------
# View loss vs its
# ------------------------

plt.figure()
plt.plot(loss_train, 'r')
plt.plot(loss_dev, 'bo--')
plt.ylabel('J')
plt.xlabel('it')
plt.grid(True)
#plt.show()
plt.savefig('losses_2L_10B_16MB.png')

# --------------------
# Save model
# --------------------

torch.save( model, model_name)


# ---------------------------------------------
# Test the model by running "testing_file.py"
# ---------------------------------------------
"""