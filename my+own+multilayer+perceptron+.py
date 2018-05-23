
# coding: utf-8

# In[4]:


#this works for regression + classification . supports linear and softmax activation funtions

import numpy as np



# In[45]:


class mlp:
    def __init__(self,inputs,targets,nhidden,beta=1,momentum=0.9,outtype="logistic"):
        self.nin=np.shape(inputs)[1]
        self.nout=np.shape(targets)[1]
        self.nhidden=nhidden
        self.ndata=np.shape(inputs)[0]
        self.beta=beta
        self.outtype=outtype
        
        self.weights1=(np.random.rand(self.nin+1,self.nhidden)-0.5)*2/np.sqrt(self.nin)
        self.weights2=(np.random.rand(self.nhidden+1,self.nout)-0.5)*2/np.sqrt(self.nhidden)
    def mlpfwd(self,inputs):
        self.hidden=np.dot(inputs,self.weights1)
        self.hidden=1.0/(1.0+np.exp(-self.beta*self.hidden))
        self.hidden=np.concatenate((self.hidden,-np.ones((np.shape(inputs)[0],1))),axis=1)
        outputs=np.dot(self.hidden,self.weights2)
        if self.outtype=="linear":
            print("output ",outputs)
            return outputs
        elif self.outtype=="logistic":
            return 1.0/(1.0+np.exp(-1*self.beta*outputs))
        elif self.outtype=="softmax":
            normalisers = np.sum(np.exp(outputs),axis=1)*np.ones((1,np.shape(outputs)[0]))
            return np.transpose(np.transpose(np.exp(outputs))/normalisers)
        else:#implement any other activation funtion
            return "error"
    
    def mlptrain(self,inputs,targets,eta,niteration=100,epsolon=0.001):
        updatew1=np.zeros((np.shape(self.weights1)))
        updatew2=np.zeros((np.shape(self.weights2)))
        inputs = np.concatenate((inputs,-np.ones((self.ndata,1))),axis=1)
        change = range(self.ndata)
        for i in range(niteration):
            #different output neurons
            self.outputs=self.mlpfwd(inputs)
            error=0.5*np.sum((self.outputs-targets)**2)
            if np.mod(i,10)==0:
                print("error ",i," th iteration",error)
            if self.outtype=="linear":

                deltao=self.outputs-self.targets/self.ndata

            elif self.outtype=="logistic":
                deltao=self.beta*(self.outputs-targets)*(self.outputs*(1.0-self.outputs))


            else:
                print("error")
            deltah=self.beta*self.hidden*(1.0-self.hidden)*np.dot(deltao,np.transpose(self.weights2))

            updatew1=eta*(np.dot(np.transpose(inputs),deltah[:,:-1]))+epsolon*updatew1    
            updatew2=eta*np.dot(np.transpose(self.hidden),deltao)+epsolon*updatew2
        
        
    def earlystopping(self,inputs,targets,valid,validtarget,eta,niteration=100,epsolon=0.001):
        pperr=10000000002
        perr=10000000001
        cerr=10000000000
        count=0
         
        valid=np.concatenate((valid,-np.ones((np.shape(valid)[0],1))),axis=1)
        
        while ((pperr-perr>epsolon)or(perr-cerr>epsolon)):
            count+=1
            print("number of times: ",count)
            self.mlptrain(inputs,targets,eta,niteration,epsolon)
            pperr=perr
            perr=cerr
            
            validout=self.mlpfwd(valid)
            cerr=0.5*np.sum((validout-validtarget)**2)
            
        print("Stopped",cerr,perr,pperr)
        return cerr
            
            
            
            
            
                
                
            
            
            
            
        
      


# In[46]:


oinputs=np.array([[1,1],[1,0],[0,1],[0,0]])
targets=np.array([[1],[1],[1],[0]])
mlpobj=mlp(oinputs,targets,nhidden=5)




# In[47]:


mlpobj.earlystopping(oinputs,targets,oinputs,targets,0.1)

