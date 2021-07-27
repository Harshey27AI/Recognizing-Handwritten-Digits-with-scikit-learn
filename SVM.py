#!/usr/bin/env python
# coding: utf-8

# In[5]:


from sklearn import svm
svc = svm.SVC(gamma=0.001, C=100.) 


# In[6]:


from sklearn import datasets
digits = datasets.load_digits()


# In[7]:


print(digits.DESCR)


# In[4]:


digits.images


# In[22]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.imshow(digits.images[1], cmap=plt.cm.copper_r, interpolation='nearest')


# In[9]:


digits.target.size


# In[26]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.subplot(321)
plt.imshow(digits.images[1791], cmap=plt.cm.gray_r,
interpolation='nearest')
plt.subplot(322)
plt.imshow(digits.images[1792], cmap=plt.cm.gray_r,
interpolation='nearest')


# In[28]:


plt.subplot(323)
plt.imshow(digits.images[1793], cmap=plt.cm.gray_r,
interpolation='nearest')
plt.subplot(324)
plt.imshow(digits.images[1794], cmap=plt.cm.gray_r,
interpolation='nearest')
plt.subplot(325)
plt.imshow(digits.images[1795], cmap=plt.cm.gray_r,
interpolation='nearest')
plt.subplot(326)
plt.imshow(digits.images[1796], cmap=plt.cm.gray_r,
interpolation='nearest')


# In[12]:


svc.fit(digits.data[1:1790], digits.target[1:1790])


# In[14]:


svc.predict(digits.data[1791:1796])


# In[ ]:




