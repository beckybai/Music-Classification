# -*- coding: utf-8 -*-
"""
Created on Sun Dec 20 17:01:31 2015

@author: Hao
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Dec 20 16:23:08 2015

@author: Hao
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 23:33:57 2015

@author: Hao
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 15:45:35 2015

@author: Hao
"""


import theano
from theano import tensor as T
import numpy as np
import scipy.io
import cPickle

from __init__ import LSTM, RNN, StackedCells, Layer, create_optimization_updates
def softmax(x):
    """
    Wrapper for softmax, helps with
    pickling, and removing one extra
    dimension that Theano adds during
    its exponential normalization.
    """
    return T.nnet.softmax(x.T)
def Relu(x):
    return T.maximum(x,0.)


def has_hidden(layer):
    """
    Whether a layer has a trainable
    initial hidden state.
    """
    return hasattr(layer, 'initial_hidden_state')

def matrixify(vector, n):
    return T.repeat(T.shape_padleft(vector), n, axis=0)

def initial_state(layer, dimensions = None):
    """
    Initalizes the recurrence relation with an initial hidden state
    if needed, else replaces with a "None" to tell Theano that
    the network **will** return something, but it does not need
    to send it to the next step of the recurrence
    """
    if dimensions is None:
        return layer.initial_hidden_state if has_hidden(layer) else None
    else:
        return matrixify(layer.initial_hidden_state, dimensions) if has_hidden(layer) else None
    
def initial_state_with_taps(layer, dimensions = None):
    """Optionally wrap tensor variable into a dict with taps=[-1]"""
    state = initial_state(layer, dimensions)
    if state is not None:
        return dict(initial=state, taps=[-1])
    else:
        return None



class Model:
    """
    Simple predictive model for forecasting words from
    sequence using LSTMs. Choose how many LSTMs to stack
    what size their memory should be, and how many
    words can be predicted.
    """
    def __init__(self,input_size=72, genre_size=10):
        # declare model
        self.model = StackedCells(input_size, celltype=LSTM, layers =[80,80],activation=lambda x:x)#*stack_size
        
# add a classifier:
        #self.model = StackedCells(input_size, celltype=celltype, layers =[],activation=Relu)
        #self.model.layers.append(Layer(33, 80, activation = Relu))
        self.model.layers.append(Layer(80,40, activation = Relu))
        self.model.layers.append(Layer(40, genre_size, activation = softmax))
        
        self.data=T.ftensor3()
        self.label=T.fmatrix()        
        self.cost=T.fscalar()
        self.res=T.fmatrix()
        # create gradient training functions:
        self.create_cost_fun()
        self.create_training_function()
        self.create_predict_function()
        
        
    @property
    def params(self):
        return self.model.params
                                 
    def create_prediction(self):
        def step(idx, *states):
            # new hiddens are the states we need to pass to LSTMs
            # from past. Because the StackedCells also include
            # the embeddings, and those have no state, we pass
            # a "None" instead:
            new_hiddens = list(states)
            
            new_states = self.model.forward(idx, prev_hiddens = new_hiddens,dropout=[0.15,None,None,None])
            return new_states
        # in sequence forecasting scenario we take everything
        # up to the before last step, and predict subsequent
        # steps ergo, 0 ... n - 1, hence:
        inputs = self.data
        num_examples = inputs.shape[1]
        # pass this to Theano's recurrence relation function:
        
        # choose what gets outputted at each timestep:
        outputs_info = [initial_state_with_taps(layer, num_examples) for layer in self.model.layers ]
#        outputs_info = [initial_state_with_taps(self.model.layers[0], num_examples),
#                        initial_state_with_taps(self.model.layers[1], num_examples)
#                        initial_state_with_taps(self.model.layers[2], num_examples)]
        result, _ = theano.scan(fn=step,
                            sequences=[inputs[:]],
                            outputs_info=outputs_info)
                                 
        self.res = result[-1][0].transpose((1,0))
        # softmaxes are the last layer of our network,
        # and are at the end of our results list:
        return self.res
                                 
    def create_cost_fun (self):
        self.create_prediction()
        self.cost = T.mean(T.nnet.categorical_crossentropy(self.res,self.label))
        #self.cost = T.sum((self.res-self.label)**2)
        self.cost_fun = theano.function(
            inputs=[self.data,self.label],
            outputs=self.cost,
            allow_input_downcast=True)
        
    def create_predict_function(self):
        self.pred_fun = theano.function(
            inputs=[self.data],
            outputs =self.res,
            allow_input_downcast=True
        )
    
                                 
    def create_training_function(self):
        updates, _, _, _, _ = create_optimization_updates(self.cost, self.params, lr=0.1, eps=1e-6, rho=0.95,method="adagrad")
        self.update_fun = theano.function(
            inputs=[self.data,self.label],
            outputs=self.cost,
            updates=updates,
            allow_input_downcast=True)
        
    def __call__(self, x):
        return self.pred_fun(x)
        
        

        
        
# construct model & theano functions:

model = Model()
print("Model Built")     

train_x = scipy.io.loadmat('Train_x5')
trX = train_x['py_data'].swapaxes(0,2).swapaxes(1,2)
train_y = scipy.io.loadmat('Train_y5')
dattrY = train_y['label']
trY = np.zeros([len(dattrY),10],np.float32)
for i in range(len(dattrY)):
    trY[i,np.int8(dattrY[i])]=1

#trX = trX[:,3000:9000,:]
#trY = trY[3000:9000]

print("Data Loaded")  
# train:
track_num = len(trY)
batchsize = 6000
batch = 24000/batchsize
error=np.zeros(batch)
acc=np.zeros(batch)
for j in range(1000):
    for i in range(batch):
        result = model.pred_fun(trX[:,batchsize*(i):batchsize*(i+1),:])
        error[i] = model.update_fun(trX[:,batchsize*(i):batchsize*(i+1),:],trY[batchsize*(i):batchsize*(i+1)]) 
        
        acc[i] = np.mean(np.argmax(trY[batchsize*(i):batchsize*(i+1)],axis=1)==np.argmax(result,axis=1))
     #   if i % 1 == 0:
        #print( error=%(error).4f, acc=%(acc).3f" % ({ "error": error[i],"acc":acc[i]}))
    #print time.time()
    test_res = result = model.pred_fun(trX[:,24000:30000,:])
    test_acc = np.mean(np.argmax(trY[24000:30000],axis=1)==np.argmax(result,axis=1))
    print("epoch %(epoch)d,error=%(error).4f,acc=%(acc).3f,test_acc=%(test_acc).3f"% ({"epoch": j,"error": np.mean(error),"acc":np.mean(acc),"test_acc":test_acc}))         
    f= file('objects.save', 'wb')
    if i%20 == 0:
        for par in model.params:
            cPickle.dump(par.get_value(borrow=True), f, -1)
        f.close()
       