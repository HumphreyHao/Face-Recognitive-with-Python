#just follow the content in the paper to get the Eigenfaces with PCA
import os
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image as im

#get the 200 neutral face,which is a.jpg
neutral = []

for i in range(201):
    img = im.open(f'dataset/{i+1}a.jpg').convert('L')
    img = img.resize((60,50), im.ANTIALIAS)
    # vectorize the img
    img2 = np.array(img).flatten() 
    neutral.append(img2)
    
img_201 = im.open(f'dataset/201a.jpg').convert('L')
img_201 = img_201.resize((60,50), im.ANTIALIAS)
img201 = np.array(img_201).flatten() 
neutral.append(img201)
plt.imshow(img201.reshape(50,60),cmap='gray'); 
plt.title('origin picture')
#cut them into 190 train set and 10 test set
#get the mean face and normalize
faces = np.vstack(neutral)
mean_face = np.mean(faces, axis=0)
faces_norm = faces - mean_face
#cut them into 190 train set and 10 test set
faces_norm_train = faces_norm[:189,:]
#calculate the covariance maxtrix, get the eigenvectors by svd
face_cov = np.cov(faces_norm_train.T)
eigen_vecs, eigen_vals, _ = np.linalg.svd(face_cov)

def mse(imageA, imageB):
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err

fig, axs = plt.subplots(2,5,figsize=(15,6))
index =199
for k, i in zip([0,1,9,19,39,79,159,199,399,799],np.arange(10)):
    # Reconstruct the first picture '20b.jpg' whose index is 19.
    weight = faces_norm[-1,:].dot(eigen_vecs[:,:k])
    projected_face = weight.dot(eigen_vecs[:,:k].T) 
    ax = plt.subplot(2,5,i+1)
    mse1=mse(projected_face.reshape(50,60),neutral[-1].reshape(50,60))
    ax.set_title("mse is "+str(mse1))
    plt.imshow(projected_face.reshape(50,60)+mean_face.reshape(50,60),cmap='gray');
fig.suptitle(("Reconstruction with Increasing Eigenfaces"), fontsize=16);
