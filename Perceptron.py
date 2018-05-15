
# coding: utf-8

# In[10]:


#simple perceptron algorithm 

import numpy as np


#    
#         

# In[42]:


class perceptron:
    
    #initialise weights
    def getweights(self,xdim,ydim):
        #+1 for bias weight
        weights=np.random.rand(xdim,ydim)*0.1-0.05
        return weights
    def __init__(self,inputs,targets):
        self.idim=np.ndim(inputs)
        self.odim=np.ndim(targets)
        self.ndata=np.shape(inputs)[0]
        #self.weights=self.getweights(self.idim,self.odim)
        
    
    
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
        self.weights=self.getweights(inputs.shape[1],1)
        for i in range(interations):
            activations=self.frwdprop(inputs)
            self.weights=self.bckprop(activations,targets,inputs,self.weights)

        print("final results ",self.frwdprop(inputs))
    def trainSeq(self,inputs,targets,interations=20,rate=0.1):
        inputs=np.concatenate((inputs,np.ones((inputs.shape[0],1))),axis=1)
        self.weights=self.getweights(inputs.shape[1],1)
        for _ in range(interations):
            for (i,t) in zip(inputs,targets):
                
                i=i.reshape((1,len(i)))
                t=t.reshape((1,len(t)))
                
                activations=self.frwdprop(i)
                self.weights=self.bckprop(activations,t,i,self.weights)
    def predict(self,inputs):
        #dding bias
        inputs=np.concatenate((inputs,np.ones((inputs.shape[0],1))),axis=1)
        print("predictions ",self.frwdprop(inputs))
        
        
    
     
     


# In[43]:


#learning XOR funtion
inputs=np.array([[0,0],[0,1],[1,0],[1,1]])
targets=np.array([[0],[1],[1],[1]])

p=perceptron(inputs,targets)
    
p.train(inputs,targets)
print("final results ",p.predict(inputs))

