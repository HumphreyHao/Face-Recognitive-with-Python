#just follow the content in the paper to get the Eigenfaces with PCA
import os
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image as im

#get the neutral face,which is a.jpg
neutral = []

for i in range(190):
    img = im.open(f'dataset/{i+1}a.jpg').convert('L')
    img = img.resize((60,50), im.ANTIALIAS)
    # vectorize the img
    img_fat = np.array(img).flatten() 
    neutral.append(img_fat)

#get the mean face and normalize
faces = np.vstack(neutral)
mean_face = np.mean(faces, axis=0)
faces_norm = faces - mean_face

#calculate the covariance maxtrix, get the eigenvectors by svd
face_cov = np.cov(faces_norm.T)
eigen_vecs, eigen_vals, _ = np.linalg.svd(face_cov)

#show 15 eigenfaces 
fig, axs = plt.subplots(1,3,figsize=(15,5))
for i in np.arange(15):
    ax = plt.subplot(3,5,i+1)
    img = eigen_vecs[:,i].reshape(50,60)
    plt.imshow(img, cmap='gray')
fig.suptitle("First 5 Eigenfaces", fontsize=16)
