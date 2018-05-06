
# coding: utf-8

# In[113]:


#simple perceptron algorithm 

import numpy as np


#    
#         

# In[120]:


class perceptron:
    

    def __init__(self,inputs,targets):
        self.idim=np.ndim(inputs)
        self.odim=np.ndim(targets)
        self.ndata=np.shape(inputs)[0]
        self.weights=getweights(self.idim,self.odim)
        
    #initialise weights
    def getweights(self,xdim,ydim):
        #+1 for bias weight
        weights=np.random.rand(xdim,ydim)*0.1-0.05
        return weights
    
    #forward propagation
    def frwdprop(self,inputfeat):
        nfeat=inputfeat.shape[1]
        nobs=inputfeat.shape[0]
        agg=np.dot(inputfeat,self.weights)
        activation=np.where(agg>0,1,0)
        return activation
    #back propagation
    def bckprop(self,activation,targets,inputs,weights,rate=0.1):
        err=np.dot(np.transpose(inputs),(activation-targets))
        weights=weights-rate*(err)
        return weights 
    def train(self,inputs,targets,interations=20,rate=0.1):
        #adding bias
        inputs=np.concatenate((inputs,np.ones((inputs.shape[0],1))),axis=1)
        weights=getweights(inputs.shape[1],1)
        for i in range(interations):
            activations=frwdprop(inputs,weights)
            weights=bckprop(activations,targets,inputs,weights)

        print("final results ",frwdprop(inputs,weights))
    
     
     


# In[97]:


#forward propagation
def frwdprop(inputfeat,weights):
    nfeat=inputfeat.shape[1]
    nobs=inputfeat.shape[0]
    agg=np.dot(inputfeat,weights)
    activation=np.where(agg>0,1,0)
    return activation

#back propagation
def bckprop(activation,targets,inputs,weights,rate=0.1):
    err=np.dot(np.transpose(inputs),(activation-targets))
    weights=weights-rate*(err)
    return weights 



def train(inputs,targets,interations=20,rate=0.1):
    #adding bias
    inputs=np.concatenate((inputs,np.ones((inputs.shape[0],1))),axis=1)
    weights=getweights(inputs.shape[1],1)
    for i in range(interations):
        activations=frwdprop(inputs,weights)
        weights=bckprop(activations,targets,inputs,weights)
    
    print("final results ",frwdprop(inputs,weights))
        


def predict(inputs,weights):
    #dding bias
    inputs=np.concatenate((inputs,np.ones((inputs.shape[0],1))),axis=1)
    print("predictions ",frwdprop(inputs,weights))


# In[122]:


#learning XOR funtion
inputs=np.array([[0,0],[0,1],[1,0],[1,1]])
targets=np.array([[0],[1],[1],[1]])

p=perceptron(inputs,targets)
    
p.train(inputs,targets)


# In[134]:


import pcn_logic_eg


# In[135]:


pt=pcn_logic_eg.pcn(inputs,targets)

