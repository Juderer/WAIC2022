import numpy as np
import torch
import torch.nn as nn
from sklearn import model_selection
from modelDesign_1 import Model_1
from modelDesign_2 import Model_2
import scipy.io as sio

DEVICE = torch.device("cpu")
# if torch.cuda.is_available():
#     DEVICE = torch.device("cuda:0")

##########################
# Model loading
model_address = 'modelSubmit_1.pth'
model_loaded = Model_1()
model_loaded.load_state_dict(torch.load(model_address, map_location='cpu'))
model_loaded.to(DEVICE)

##########################
# Case 1
file_name1 = '../dataset/data/Case_1_2_Training.npy'
CIR = np.load(file_name1)
trainX = CIR.transpose((2, 1, 3, 0))[:3000]  # [none, 256, 72, 2]
Num = trainX.shape[0]  # Number of samples

file_name2 = '../dataset/data/Case_1_2_Training_Label.npy'
POS = np.load(file_name2)
trainY = POS.transpose((1, 0))[:3000]

trainX = torch.from_numpy(trainX.astype(np.float32)).to(DEVICE)
y_test = model_loaded(trainX).cpu().data.numpy()
Diff = np.sqrt(np.sum((np.square(y_test - trainY)), 1))  # Euclidean distance
Order = np.sort(Diff)
Score_Case1 = Order[int(Num * 0.9)]

##########################
# Case 2
# file_name1 = '../dataset/data/Case_3_Training.npy'
# CIR = np.load(file_name1)
# trainX = CIR.transpose((2, 1, 3, 0))  # [none, 256, 72, 2]
# Num = trainX.shape[0]  # Number of samples
#
# file_name2 = '../dataset/data/Case_3_Training_Label.npy'
# POS = np.load(file_name2)
# trainY = POS.transpose((1, 0))
#
# trainX = torch.from_numpy(trainX.astype(np.float32)).to(DEVICE)
# y_test = model_loaded(trainX).cpu().data.numpy()
# Diff = np.sqrt(np.sum((np.square(y_test - trainY)), 1))  # Euclidean distance
# Order = np.sort(Diff)
# Score_Case2 = Order[int(Num * 0.9)]

##########################
# Model loading
model_address = 'modelSubmit_2.pth'
model_loaded = Model_2()
model_loaded.load_state_dict(torch.load(model_address, map_location='cpu'))
model_loaded.to(DEVICE)

##########################
# Case 3
file_name1 = '../dataset/data/Case_3_Training.npy'
CIR = np.load(file_name1)
trainX = CIR.transpose((2, 1, 3, 0))[:1000]  # [none, 256, 72, 2]
Num = trainX.shape[0]  # Number of samples

file_name2 = '../dataset/data/Case_3_Training_Label.npy'
POS = np.load(file_name2)[:1000]
trainY = POS.transpose((1, 0))

trainX = torch.from_numpy(trainX.astype(np.float32)).to(DEVICE)
y_test = model_loaded(trainX).cpu().data.numpy()
Diff = np.sqrt(np.sum((np.square(y_test - trainY)), 1))  # Euclidean distance
Order = np.sort(Diff)
Score_Case3 = Order[int(Num * 0.9)]

##########################
# Final print
print('The score of Case 1 is ' + np.str(Score_Case1))
# print('The score of Case 2 is ' + np.str(Score_Case2))
print('The score of Case 3 is ' + np.str(Score_Case3))

# score = 100 - (Score_Case1 + Score_Case2 + Score_Case3)
score = 100 - (Score_Case1 + Score_Case3)
print('The final score is ' + np.str(score))

print('END')
