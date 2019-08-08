import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import pathlib
from PIL import Image
from scipy.linalg import eigh
import re

img_paths = pathlib.Path('drive/My Drive/imglq/train/').glob('*')
img_sorted = sorted([x for x in img_paths])

from google.colab import drive
drive.mount('/content/drive')

paths = list(map(str, img_sorted))
names_train = [re.findall(r'/([^/]+)_\d+.*$',p)[0] for p in paths]

img = []
for im_path in img_sorted[:]:
    im = np.array(Image.open(str(im_path)).convert('L'))
    img.append(np.array(im.reshape(-1)))

data = np.array(img).T
mean = np.mean(data,axis = 1).reshape(-1,1)
data = data - mean

data.shape

print(len(img))
print(data.shape)

cov = np.cov(data.T)
lam, V = eigh(cov,eigvals=(19,23))

print(cov.shape)
print(V.shape)
lam.shape

eigen_face = V.T @ data.T
eigen_face.shape

signature = (eigen_face @ data)

signature.shape

mean_proj = np.mean(signature,axis = 1)
mean_proj

def find_mean_class(Signature):
    mean_class = []
    for i in range(6):
        mean_class.append(Signature[:,4*(i-1):4*i-1].mean(axis=1))
    return np.array(mean_class)
mean_class = find_mean_class(signature)
mean_class.shape


def find_Scatter_within_class(Signature):
    Scatter_within_class = np.zeros((Signature.shape[0], Signature.shape[0]))
    
    for i in range(6):
        Scatter_within_class = Scatter_within_class + Signature[:,4*(i-1):4*i-1] @ Signature[:,4*(i-1):4*i-1].T
    return Scatter_within_class
Scatter_within_class = find_Scatter_within_class(signature)


Scatter_within_class.shape

def find_Scatter_between_class(mean_class, Mean_Projected_faces):
    sb = np.zeros((mean_class.shape[1],mean_class.shape[1]))
    
    for i in range(mean_class.shape[0]):
        mean_class[i,:] = mean_class[i,:] - Mean_Projected_faces
        m = mean_class[i,:].reshape(mean_class.shape[1], 1)
        sb = sb + (m @ mean_class[i,:].reshape(mean_class.shape[1], 1).T)
    
    return sb

Scatter_between_class = find_Scatter_between_class(mean_class, mean_proj)

Scatter_between_class.shape

# Criterion_matrix
J = np.linalg.inv(Scatter_within_class) @ Scatter_between_class

J.shape

eig_vals, eig_vecs = eigh(J)
eig_vals, eig_vecs.shape

eig_vecs.shape

w = eig_vecs[:, 2:]
w.shape

# Fishers face (FF)
FF = w.T @ signature

FF.shape

test_paths = pathlib.Path('drive/My Drive/imglq/test/').glob('*')
test_sorted = sorted([x for x in test_paths])

paths = list(map(str, test_sorted))
names_test = [re.findall(r'/([^/]+)_\d+.*$',p)[0] for p in paths]
names_test

test_img = []
for im_path in test_sorted[:]:
    im = np.array(Image.open(str(im_path)).convert('L'))
    test_img.append(np.array(im.reshape(-1)))
test_img = np.array(test_img).T - mean

test_img.shape

eigen_face.shape

proj = eigen_face @ test_img

proj.shape

Projected_Fisher_Test_Img = w.T @ proj

Projected_Fisher_Test_Img.shape


print('Actual','\t\tPredicted')
print('----'*8)
for i in range(6):
    t = Projected_Fisher_Test_Img[:,i]
    preds = np.argsort(np.sum((FF - t.reshape(-1,1)) ** 2,axis=0))[23:20:-1]
    print(names_test[i], end='\t\t')
    for j in preds:
        print(names_train[j], end='\t')
    print()
 
 t = proj[:,0]
np.argsort(np.sum((signature - t.reshape(-1,1)) ** 2,axis=0))[23:20:-1]

signature.shape
