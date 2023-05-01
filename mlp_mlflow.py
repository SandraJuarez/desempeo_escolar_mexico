import jax
import jax_metrics as jm
import jax.numpy as jnp
from jax import grad, jit, vmap
from functools import partial
from jax import random
import os
import numpy as np
import matplotlib.pyplot as plt# Switch off the cache 
from sklearn.metrics import confusion_matrix 
import mlflow
import metrics
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'
class MultilayerPerceptron():
    def __init__(self,layers,lr,labels,k_clases,samples,x,y,y_hot,stop,max_steps):
        ''' 
        layers=[input,hidden,...,hidden,output]
        lr=learning rate
        epochs
        labels= [0,1,2,3,...k]
        k_clases=k
        samples=n

        '''
        self.sizes=layers #[input, hidden, hidden,....,output]
        self.lr=lr
        self.labels=labels
        self.k_clases=k_clases
        self.samples=samples
        self.x=x
        self.y=y
        self.y_hot=y_hot
        self.stop=stop #stopping criteria
        self.max_steps=max_steps #maximum number of steps
        
        self.weights=self.init_network_params(self.sizes, random.PRNGKey(0) )


    #tenemos la matriz de pesos. Como tenemos la capa de input y una hidden, por eso tenemos dos pessos y dos bias    
    @staticmethod
    def random_layer_params(m, n, key, scale=1):
        w_key, b_key = random.split(key)
        return scale * random.normal(w_key, (n, m)), scale * random.normal(b_key, (n,1))

        # Initialize all layers for a fully-connected neural network with sizes "sizes"
    def init_network_params(self,sizes, key):
        keys = random.split(key, len(sizes))
        return [self.random_layer_params(m, n, k) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]
    #@staticmethod
    def one_hot(y, k_clases):
        """Create a one-hot encoding of y of size k_clases."""
        return jnp.array(y[:, None] == jnp.arange(k_clases))
    @staticmethod
    def activacion(x):
        return jnp.tanh(x)#jnp.maximum(0, x)#
    @staticmethod
    def softmax(Z):
        A = jnp.exp(Z) / jnp.sum(jnp.exp(Z),axis=0)
        return A

    def forward(self,weights,x):
        #input to hidden layers
        activations=x
        for w,b in weights[:-1]:
            outputs=jnp.dot(w,activations) + b #size -> (hidden,hidden anterior)
            activations=self.activacion(outputs)
            
        #last hidden to output
        #we use softmax for the last one
        w_last, b_last = weights[-1]
        #print(w_last)
        logits = jnp.dot(w_last, activations) + b_last
        #print('logits',logits)
        soft=self.softmax(logits) #size -> (classes,samples) ****
        return soft
    
    def loss_function(self,weights,x,y_hot):

        soft=self.forward(weights,x)
        #print('soft',soft)
        #loss=jnp.mean(-y_hot*jnp.log(soft))
        
        return jnp.mean(-y_hot*jnp.log(soft))
        
    #@partial(jit, static_argnums=(0,))
    def update( self,weights, x, y,lr):
        grads = grad(self.loss_function)(weights, x, y)
        
        return [(w - lr * dw ,b - lr* db)
                        for (w, b), (dw, db) in zip(weights, grads)]

    def get_accuracy(self,predictions, Y):
        #print(predictions, Y)
        return jnp.sum(predictions == Y) / Y.size

    
    def prediction(self,soft): 
        return jnp.argmax(soft,axis=0)

    def get_pr(self,k_classes,samples,clases,y0,y_hat):
        FP=0
        FN=0
        TP=0
        recall_list=[]
        samples=100
        precision_list=[]
        for k in range(k_classes):
            for i in range (samples):
                if y0[i]==clases[k] and y_hat[i]==clases[k]:
                    TP+=1
                if y0[i]!=clases[k] and y_hat[i] == clases[k]:
                    FP+=1
                if y0[i]==clases[k] and y_hat[i] != clases[k]:
                    FN+=1
            if FP+TP!=0:
                precision_list.append(TP/(TP+FP))
            else:
                precision_list.append(0)
            if FN+TP!=0:
                recall_list.append(TP/(TP+FN))
            else:
                recall_list.append(0)

            
            
    
        precision=sum(precision_list)/k_classes
        recall=sum(recall_list)/k_classes
        return precision,recall


    

    def modelo(self,weights,max_steps,x,y,y_hot,learning_rate,k_clases,samples,labels,stop,x_val,y_val,y_hot_val,samples_val,run_name):
        experiment_name = "multilayer"
        experiment = mlflow.get_experiment_by_name(experiment_name)
        experiment_id = experiment.experiment_id
        #y_hot=one_hot(y, k_clases)
        #the initial parameters
        #params=init_network_params(sizes,random.PRNGKey(0))
        with mlflow.start_run(experiment_id= experiment_id, run_name=run_name) as run:
            features=x.shape[0]
            arquitectura=str(self.sizes)
            mlflow.log_param('features',features)
            mlflow.log_param('arquitectura',arquitectura)
            mlflow.log_param('learning_rate',learning_rate)
            #mlflow.log_param('capas',self.layers)
            precision_list=[]
            recall_list=[]
            loss_list=[]
            precision_list_val=[]
            recall_list_val=[]
            loss_list_val=[]
            loss=1000
            
            for i in range(max_steps):
                old_loss=loss
                loss=self.loss_function(weights,x,y_hot)
                #print(loss)
                
                weights=self.update(weights, x, y_hot,learning_rate)
                
                if i%10==0:
                    soft=self.forward(weights,x)
                    soft_val=self.forward(weights,x_val)
                    y_hat=self.prediction(soft)
                    y_hat_val=self.prediction(soft_val)
                    ac=self.get_accuracy(y_hat, y)
                    ac_val=self.get_accuracy(y_hat_val,y_val)
                    loss_val=self.loss_function(weights,x_val,y_hot_val)
                    loss_list.append(loss)
                    loss_list_val.append(loss_val)
                    print('TRAINING: In the epoch {:5d} the loss is {:2.5f} and the accuracy is {:2.5f}'.format(i,loss,ac))
                    print('VALIDATION: In the epoch {:5d} the loss is {:2.5f} and the accuracy is {:2.5f}'.format(i,loss_val,ac_val))
                if i%200==0:
                    precision,recall=metrics.precision(y,y_hat),metrics.recall(y,y_hat)
                    precision_list.append(precision)
                    recall_list.append(recall)
                    precision_val,recall_val=metrics.precision(y_val,y_hat_val),metrics.recall(y_val,y_hat_val)
                    precision_list_val.append(precision_val)
                    recall_list_val.append(recall_val)
                    print('TRAINING:In the epoch {:5d} the precission is {:2.5f} and the recall is {:2.5f}'.format(i,precision,recall))
                    print('VALIDATION:In the epoch {:5d} the precission is {:2.5f} and the recall is {:2.5f}'.format(i,precision_val,recall_val))
                    
                    
                if  jnp.abs(loss-old_loss)<stop:
                    precision,recall=metrics.precision(y,y_hat),metrics.recall(y,y_hat)
                    precision_val,recall_val=metrics.precision(y_val,y_hat_val),metrics.recall(y_val,y_hat_val)
                    precision_list.append(precision)
                    recall_list.append(recall)
                    precision_list_val.append(precision_val)
                    recall_list_val.append(recall_val)
                    print('At the final epoch {:5d} the precission is {:2.5f} and the recall is {:2.5f}'.format(i,precision,recall))
                    break
                #print(i,params)
                #print(loss)
            mlflow.log_metric('training_loss',loss)
            mlflow.log_metric('precision training',precision)
            mlflow.log_metric('recall training',recall)
            mlflow.log_metric('validation_loss',loss_val)
            mlflow.log_metric('validation_precission',precision_val)
            mlflow.log_metric('validation_recall',precision_val)
            
        return loss,recall_list,precision_list,loss_list,precision_list_val,recall_list_val,loss_list_val


    def modelo_boosting(self,weights,max_steps,x,y_hot,learning_rate,stop,x_v,y_v):
            
            precision_list=[]
            recall_list=[]
            loss_list=[]
            precision_list_val=[]
            recall_list_val=[]
            loss_list_val=[]
            loss=1000
            
            for i in range(max_steps):
                old_loss=loss
                loss=self.loss_function(weights,x,y_hot)
                #print(loss)
                
                weights=self.update(weights, x, y_hot,learning_rate)
                
                if i%10==0:
                    soft=self.forward(weights,x)
                    soft_val=self.forward(weights,x_v)
                    y_hat=self.prediction(soft)
                    y_hat_val=self.prediction(soft_val)
                    
                    
                    
                if  jnp.abs(loss-old_loss)<stop:
                    print('End of an mlp')
                    break
                #print(i,params)
                #print(loss)
            
            return y_hat,y_hat_val
    
    if __name__=="__MAIN__":
        modelo()
        modelo_boosting()
