
# coding: utf-8

# In[7]:



import numpy as np


# In[9]:



def LeastSquareOpt(lInputs,ltargets):
    inputbias=np.concatenate((lInputs,-np.ones((lInputs.shape[0],1))),axis=1)
    beta=np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(inputbias),inputbias)),np.transpose(inputbias)),ltargets)
    predictions=np.dot(inputbias,beta)
    return predictions


    
    


# In[10]:


#exapmle using

inputs=np.array([[0,0],[0,1],[1,0],[1,1]])
targets=np.array([[0],[1],[1],[1]])

LeastSquareOpt(inputs,targets)

