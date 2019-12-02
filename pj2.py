#just follow the content in the paper to get the Eigenfaces with PCA
import os
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image as im

#define the MSE calculate function
def mse(imageA, imageB):
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err

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
#plot the MSE with different PCs
fig, axs = plt.subplots(2,5,figsize=(15,6))
index =19
for k, i in zip([0,4,9,29,49,69,99,139,159,189],np.arange(10)):
    # Reconstruct the picture '20a.jpg' whose index is 19.
    weight = faces_norm[index,:].dot(eigen_vecs[:,:k])
    projected_face = weight.dot(eigen_vecs[:,:k].T) 
    ax = plt.subplot(2,5,i+1)
    mse1=mse(projected_face.reshape(50,60)+mean_face.reshape(50,60),neutral[index].reshape(50,60))
    ax.set_title("K is "+ str(k)+ " mse is "+str(int(mse1)))
    plt.imshow(projected_face.reshape(50,60)+mean_face.reshape(50,60),cmap='gray');
fig.suptitle(("Reconstruction with Increasing Eigenfaces"), fontsize=16);
