#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ipywidgets as widgets
from ipywidgets import interact

from sklearn.decomposition import PCA

# use seaborn plotting style defaults
import seaborn as sns; sns.set()

from plot_utils import *


# In[2]:


plt.rcParams['figure.figsize'] = (10, 8)


# 
# Principal Component Analysis (PCA)
# 
# For a complete introduction, please see Dimensionality Reduction
# 
# Here we'll explore Principal Component Analysis, which is an
# 
# extremely useful linear dimensionality reduction technique.
# 
# The goal is to reduce the size (dimensionality) of a dataset while capturing most of its information.
# 
# There are many reason why dimensionality reduction can be useful:
# 
# It can reduce the computational cost when running learning
# algorithms,
# 
# decrease the storage space, and
# 
# may help with the so-called "curse of dimensionality,"
# 

# In[3]:


# create random data from normal distribution
np.random.seed(1)
X = np.dot(np.random.random(size=(2, 2)), np.random.normal(size=(2, 200))).T

# plot data
c = PCA().fit_transform(X)[:, 0]  # colors
plt.scatter(X[:, 0], X[:, 1], s=50, c=c, cmap='viridis')
plt.title('Original Data')
plt.axis('equal')
plt.show()


# In[5]:


pca =PCA(n_components = 2).fit(X)


# In[6]:


U = pca.components_          # Principal Components (directions)
S = pca.explained_variance_  # importance of ecah direction (variances)

print("1st Principal Component: {} ({:.2f})".format(U[0], S[0]))
print("2nd Principal Component: {} ({:.2f})".format(U[1], S[1]))


# In[7]:


print(np.linalg.norm(U[0]))
print(np.linalg.norm(U[1]))


# Properties
# 
# Matrix U is an orthogonal matrix:
# 
# An orthogonal matrix is a square matrix whose columns and rows are orthogonal unit vectors.

# In[9]:


print(np.dot(U[0], U[1]))


# In[10]:


"""To see what these numbers mean, let's view them as
vectors plotted on top of the data:
"""

# plot data
plt.scatter(X[:, 0], X[:, 1], s=50, c=c, cmap='viridis', alpha=0.5)

plt.arrow(0, 0, 3 * np.sqrt(S[0]) * U[0, 0], 3 * np.sqrt(S[0]) * U[0, 1], width=.03, head_width=.1, color='k')
plt.arrow(0, 0, 3 * np.sqrt(S[1]) * U[1, 0], 3 * np.sqrt(S[1]) * U[1, 1], width=.03, head_width=.1, color='k')

plt.title("Principal Components")
plt.axis('equal')
plt.show()


# """Notice that one vector is longer than the other,
# which means that direction in the data is somehow more
# important
# than the other direction.
# 
# In other word, the second principal component could be completely
# ignored without much loss of information! Let's see what our
# data look like if we only keep 95% of the variance:"""

# In[11]:


pca =PCA (0.95 ) #keep 95% of  varince
X_proj = pca.fit_transform(X)
print(X.shape)
print(X_proj.shape)

"""By specifying that we want to throw away 5% of the variance,
the data is now compressed by a factor of 50%! Let's see what
the data look like after this compression:
"""


# In[12]:


X_approx = pca.inverse_transform(X_proj)

plt.scatter(X[:, 0], X[:, 1], s=50, c=c, cmap='viridis', alpha=0.2)
#plot orjnal data
plt.scatter(X_approx[:, 0], X_approx[:, 1], s=50, c=c, cmap='viridis', alpha=0.9)  # plot projected data

plt.title("projected Data")
plt.axis('equal')
plt.show()

"""e light points are the original data, while the dark points are the
projected version.
We see that after truncating 5% of the variance of this
dataset and then reprojecting it,
the "most important" features of the data are maintained,
and we've compressed the data by 50%!

This is the sense in which "dimensionality reduction"
works: if you can approximate a
data set in a lower 
dimension,
you can often have an easier time visualizing it or fitting
complicated models to
the data.
"""


# In[13]:


from sklearn.datasets import load_digits
digits = load_digits()
X,y = digits.data,digits.target


# """Perform PCA to project data from 64 to 2 dimensions:
# """

# In[14]:


pca = pca = PCA(n_components=2)  
X_proj = pca.fit_transform(X)
#project from  64 to 2 dimensions
print("shape of original data: {}".format(X.shape))
print("shape of projected data:{}".format(X_proj.shape))


# In[15]:


plt.scatter(X_proj[:, 0],X_proj[:, 1],c=y,edgecolors='none',
            cmap=plt.cm.get_cmap('nipy_spectral', 10))
plt.colorbar()  
plt.show()


# """
# This gives us an idea of the relationship between the digits.
# Essentially, we have found the optimal stretch and rotation in 64-dimensional 
# space that allows us to see the layout of the digits,
# without reference to the labels
# 
# 
# What do the Components Mean?
# 
# PCA is a very useful dimensionality reduction algorithm, because it has a very intuitive interpretation via eigenvectors. The input data is represented as a vector: in the case of the digits, our data is
# 
# but what this really means is
# 
# $$
# image(x) = x_1 \cdot{\rm (pixel~1)} + x_2 \cdot{\rm (pixel~2)} + x_3 \cdot{\rm (pixel~3)} \cdots
# $$
# If we reduce the dimensionality in the pixel space to (say) 6, we recover only a partial image:
# 
# """
# 

# In[16]:


plt.matshow(digits.images[0]);


# """But the pixel-wise representation is not the only choice.
# We can also use other basis functions, and write something like
# 
# $$
# image(x) = {\rm mean} + x_1 \cdot{\rm (basis~1)} + x_2 \cdot{\rm (basis~2)} + x_3 \cdot{\rm (basis~3)} \cdots
# $$
# What PCA does is to choose optimal basis functions so that only a few are needed to get a reasonable 
# approximation. 
# The low-dimensional representation of our data is the coefficients of this series, and the approximate
# reconstruction
# is the result of the sum.
# 

# In[17]:


plt.figure(figsize=(12, 6))
pca = PCA().fit(X)

plt.bar(range(len(pca.explained_variance_ratio_)),
pca.explained_variance_ratio_,alpha=0.5,align='center')

plt.step(range(len(pca.explained_variance_ratio_)),
         np.cumsum(pca.explained_variance_ratio_),
         where='mid')

plt.xlabel('number of components');
plt.ylabel('cumulative explained varince');


# """
# Choosing the Number of Components
# But how much information have we thrown away?
# We can figure this out by looking at the explained
# variance as a function of the components
# """

# In[18]:


total_var = np.cumsum(pca.explained_variance_ratio_)
for i in [0, 1, 2, 3, 4, 5, 8, 12, 20 ,27 ,36 ,40 ,50]:
    print("components:{:2d},total explained varaince{:.2f}".format(i, total_var[i]))


# """
# 
# Here we see that our two-dimensional projection loses a lot of information 
# (as measured by the explained variance) and that we'd need about 20 components
# to retain 90% of the variance. Looking at this plot for a high-dimensional dataset
# can help you understand the level of redundancy present in multiple observations.
# """

# In[19]:


K = 30
plt.figure(figsize=(12, 6))

pca = PCA().fit(X,y)  # Notice

plt.bar(range(K),
        pca.explained_variance_ratio_[:K],
        alpha=0.5,
        align='center')

plt.step(range(K),
         np.cumsum(pca.explained_variance_ratio_[:K]),
         where='mid')

plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.show()
        


# In[20]:



total_var = np.cumsum(pca.explained_variance_ratio_[:K])

for i in [0, 1, 2, 3, 4, 5, 10, 15, 20, 29]:
    print("Components: {:2d}, total explained variance: {:.2f}".format(i, total_var[i]))


# """
# 
# PCA as data compression:
# As we mentioned, PCA can be used for is a sort of data compression.
# Using a small n_components allows you to represent a high dimensional
# point as a sum of just a few principal vectors.
# Here's what a single digit looks like as you change
# the number of components:
# """

# In[21]:


def update_pca_plot(i, n_components):
    pca = PCA(n_components).fit(X)
    im = pca.inverse_transform(pca.transform(X[i:i+1]))
    
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.imshow(im.reshape((8, 8)), cmap='binary')
    plt.title('Approximated Data (k={})'.format(n_components))
    plt.axis('off')

    plt.subplot(122)
    plt.imshow(X[i].reshape((8, 8)), cmap='binary')
    plt.title('Original Data')
    plt.axis('off')
    plt.show()


idx = widgets.IntSlider(value=20, min=0, max=1796, desc='data')
interact(update_pca_plot, i=idx, n_components=range(1, 65));


# In[22]:


def show_all_digit_components(X, index=None):
    index = np.random.choice(X.shape[0]) if index is None else index
    
    fig, axes = plt.subplots(8, 8, figsize=(10, 10))
    fig.subplots_adjust(hspace=0.1, wspace=0.1)

    for i, ax in enumerate(axes.flat):
        pca = PCA(i + 1).fit(X)
        im = pca.inverse_transform(pca.transform(X[index:index+1]))

        ax.imshow(im.reshape((8, 8)), cmap='binary')
        ax.text(0.95, 0.05, '{0}'.format(i + 1), ha='right', transform=ax.transAxes, color='r')
        ax.set_xticks([])
        ax.set_yticks([])
        
    plt.show()
    

show_all_digit_components(X)


# Let's take another look at this by using IPython's interact functionality to view the reconstruction of several images at once:

# In[23]:


def plot_digits(n_components):
    fig = plt.figure(figsize=(16, 8))
    nside = 10
    
    pca = PCA(n_components).fit(X)
    X_proj = pca.inverse_transform(pca.transform(X[:nside ** 2]))
    X_proj = np.reshape(X_proj, (nside, nside,8, 8))
    total_var = pca.explained_variance_ratio_.sum()
    
    plt.subplot(121)
    im = np.vstack([np.hstack([X_proj[i, j] for j in range(nside)])
                    for i in range(nside)])
    plt.imshow(im, cmap='binary')
    plt.title("k = {0}, variance = {1:.2f}".format(n_components, total_var), size=18)
    plt.axis('off')
    plt.clim(0, 1);
    
    plt.subplot(122)
    X_org = X[:nside ** 2].reshape((nside, nside,8 ,8))
    im = np.vstack([np.hstack([X_org[i, j] for j in range(nside)]) for i in range(nside)])
    plt.imshow(im, cmap='binary')
    plt.title("Original Data", size=18);
    plt.axis('off');
    

interact(plot_digits, n_components=[10, 20, 30, 40, 50, 100, 150, 200, 784]);


# In[24]:


from sklearn.datasets import fetch_olivetti_faces

faces = fetch_olivetti_faces()
X, y = faces['data'], faces['target']
print(X.shape)


# In[25]:


# select 100 faces randomly
X_samples = np.random.permutation(X)[:700]

fig, axes = plt.subplots(12, 12, figsize=(14, 14))
fig.subplots_adjust(left=0.2, bottom=0.2, right=0.8, top=0.8, hspace=0.01, wspace=0.01)

for i, ax in enumerate(axes.flat):
    ax.imshow(X_samples[i].reshape((64, 64)), cmap='gray')
    ax.set_xticks(())
    ax.set_yticks(())
plt.show()


# In[26]:


def update_pca_face_plot(i, n_components):
    pca = PCA(n_components).fit(X)
    im = pca.inverse_transform(pca.transform(X[i:i+1]))
    
    fig = plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.imshow(im.reshape((64, 64)), cmap='gray')
    total_var = pca.explained_variance_ratio_.sum()
    plt.title('Approximated Data ({:.2f})'.format(total_var))
    plt.axis('off')

    plt.subplot(122)
    plt.imshow(X[i].reshape((64, 64)), cmap='gray')
    plt.title('Original Data')
    plt.axis('off')
    plt.show()


idx = widgets.IntSlider(value=32, min=0, max=400, desc='data')
widgets.interact(update_pca_face_plot, i=idx, n_components=[10, 50, 100, 150, 200, 300, 400]);


# In[28]:


nside=5
X_samples = np.random.permutation(X)[:nside ** 2]


def plot_faces(n_components):
    global X_samples
    
    fig = plt.figure(figsize=(12, 6))    
    pca = PCA(n_components).fit(X)
    X_proj = pca.inverse_transform(pca.transform(X_samples))
    X_proj = np.reshape(X_proj, (nside, nside, 64, 64))
    total_var = pca.explained_variance_ratio_.sum()
    
    
    
    plt.subplot(121)
    im = np.vstack([np.hstack([X_proj[i, j] for j in range(nside)]) for i in range(nside)])
    plt.imshow(im, cmap='gray')
    plt.axis('off')
    plt.title("k = {0}, variance = {1:.2f}".format(n_components, total_var), size=18)
    
    plt.subplot(122)
    X_org = np.reshape(X_samples, (nside, nside, 64, 64))
    im = np.vstack([np.hstack([X_org[i, j] for j in range(nside)]) for i in range(nside)])
    plt.imshow(im, cmap='gray')
    plt.axis('off')
    plt.title("Original Faces", size=18)
    

interact(plot_faces, n_components=[10, 20, 30, 40, 50, 100, 150, 200]);


# In[29]:


def plot_faces_components(size):
    n_components = size ** 2
    pca = PCA(n_components).fit(X)
    C = np.reshape(pca.components_[:n_components], (size, size, 64, 64))
    total_var = pca.explained_variance_ratio_.sum()
    
    
    im = np.vstack([np.hstack([C[i, j] for j in range(size)]) for i in range(size)])

    fig = plt.figure(figsize=(10, 10))
    plt.imshow(im, cmap='gray')
    plt.xticks(())
    plt.yticks(())
    plt.title("k = {0}, variance = {1:.2f}".format(n_components, total_var), size=18)
    plt.show()



size = widgets.IntSlider(value=5, min=1, max=64, step=1, desc='size')
interact(plot_faces_components, size=size);


# In[30]:


def plot_faces_components(size, index):
    n_components = size ** 2
    pca = PCA(n_components).fit(X)
    C = np.reshape(pca.components_[:n_components], (size, size, 64, 64))
    total_var = pca.explained_variance_ratio_.sum()
    
    
    im = np.vstack([np.hstack([C[i, j] for j in range(size)]) for i in range(size)])
    
    x_proj = pca.transform(X[index:index+1])
    x_approx = pca.inverse_transform(x_proj)
    fig = plt.figure(figsize=(18, 6))
    plt.subplot(131)
    plt.imshow(im, cmap='gray')
    plt.xticks(())
    plt.yticks(())
    plt.title("Principal Components".format(n_components, total_var), size=18)
                     
    plt.subplot(132)
    plt.imshow(x_approx.reshape(64, 64), cmap='gray')
    plt.xticks(())
    plt.yticks(())
    plt.title("k = {0}, variance = {1:.2f}".format(n_components, total_var), size=18)

    plt.subplot(133)
    plt.imshow(X[index].reshape(64, 64), cmap='gray')
    plt.xticks(())
    plt.yticks(())
    plt.title("Original face", size=18)
    plt.show()    
size = widgets.IntSlider(value=1, min=1, max=20, step=1, desc='size')
index = widgets.IntSlider(value=32, min=0, max=399, step=1, desc='index')
interact(plot_faces_components, size=size, index=index);


# In[37]:


from numpy.linalg import svd
np.set_printoptions(precision=2)


# In[38]:


def PCA(X, k=2):
    """ Principal Component Analysis implementation
    
    Arguments:
        - X: data matrix - numpy array of shape (m, n)
        - k: number of components
        
    Returns:
       - Projection of X into a k-d space of principal components
    
    """
    m = X.shape[0]
    
    Xn = X - X.mean(axis=0)   # STEP 1: zero-center data (remove mean)          
    Sigma = (Xn.T @ Xn) / m   # STEP 2: compute covariance matrix
    U, S, VT = svd(Sigma)     # STEP 3: Singular Value Decomposition
    
    X_proj = Xn @ U[:, :k]    # project data
    return X_proj


# In[39]:


X = np.array([[1, 1, 1, 0, 0], 
              [2, 2, 2, 0, 0], 
              [1, 1, 1, 0, 0], 
              [5, 5, 5, 0, 0], 
              [1, 1, 0, 2, 2], 
              [0, 0, 0, 3, 3], 
              [0, 0, 0, 1, 1]], dtype=np.float32)
print(X)


# In[40]:


X_proj = PCA(X, k=3)
print(X_proj)


# STEP 1: Zero-center data

# In[41]:


mu = np.mean(X,axis=0)
X_norm = X-mu
print(X_norm)


# STEP 2: Compute covariance matrix

# In[42]:


m = X.shape[0]
Sigma = (X_norm.T @ X_norm) / m


# STEP 3: Singular Value Decomposition

# In[43]:


U, S, V = svd(Sigma)


# In[44]:


print(U)


# In[45]:


print(S)


# Project data:

# In[46]:


X_proj = X_norm @ U [: , : 3]
print(X_proj)


# Recover data:

# In[47]:


X_approx = X_proj @ U [:, : 3].T + mu
print(X_approx)


# In[48]:


print(X)


# Choosing number of principal components:

# In[50]:


m, n = X.shape

Xn = X - X.mean(axis=0)
Sigma = (Xn.T @ Xn) / m
U, S, V = svd(Sigma)

for k in range(1, n + 1):
    total_var = np.sum(S[:k]) / np.sum(S)
    print("k = {:d}, explained variance = {:.3f}".format(k, total_var))
    if total_var >= 0.99: break


# Appendix: Singular Value Decomposition

# In[51]:


X = np.array([[1, 1, 1, 0, 0], 
              [2, 2, 2, 0, 0], 
              [1, 1, 1, 0, 0], 
              [5, 5, 5, 0, 0], 
              [1, 1, 0, 2, 2], 
              [0, 0, 0, 3, 3], 
              [0, 0, 0, 1, 1]], dtype=np.float32)


# In[52]:


U , sigma, VT =svd(X)


# In[53]:


print(sigma)


# In[54]:


np.diag(sigma)


# In[55]:


print(U.shape,VT.shape)


# In[56]:


X_approx = U[: , : 1] @ np.diag(sigma)[: 1, : 1] @ VT[: 1, :]
print("SSE = {:.2f}".format(np.linalg.norm(X - X_approx) **2))
print(X_approx)


# In[57]:


X_approx = U[: , : 2] @ np.diag(sigma)[: 2, : 2] @ VT[: 2, :]
print("SSE = {:.2f}".format(np.linalg.norm(X - X_approx)** 2))
print(X_approx)


# In[58]:


X_approx = U[: , : 3] @ np.diag(sigma)[: 3, : 3] @ VT[: 3, :]
print("SSE = {}".format(np.linalg.norm(X - X_approx) ** 2))
print(X_approx)


# In[59]:


X_approx = U[: , : 4] @ np.diag(sigma)[: 4, : 4] @ VT[: 4, :]
print("SSE = {}".format(np.linalg.norm(X - X_approx) ** 2))
print(X_approx)


# In[60]:


X_approx = U [: , : 5] @  np.diag(sigma)[: 5, : 5] @ VT[:5, :]
print("SSE{}".format(np.linalg.norm(X - X_approx) **3))
print(X_approx)


# In[ ]:




