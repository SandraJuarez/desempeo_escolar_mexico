import jax
import jax_metrics as jm
import jax.numpy as jnp
from jax import grad, jit, vmap
from functools import partial
from jax import random
import os
import numpy as np
import matplotlib.pyplot as plt
import mlflow
import graficar
import importlib
import metrics
# Switch off the cache 
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'

class Linear_Model():
    """
    Basic Linear Regression with Ridge Regression
    """
    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.key = random.PRNGKey(0)
        self.cpus = jax.devices("cpu")

    @staticmethod
    @jit
    def linear_model(X: jnp, theta: jnp) -> jnp:
        """
        Classic Linear Model. Jit has been used to accelerate the loops after the first one
        for the Gradient Descent part
        args:
            X: Data array at the GPU or CPU
            theta: Parameter w for weights and b for bias
        returns:
            f(x): the escalar estimation on vector x or the array of estimations
        """
        w = theta[:-1]
        b = theta[-1]
        return jax.numpy.matmul(X, w) + b

    def generate_theta(self):
        """
        Use the random generator at Jax to generate a random generator to instanciate
        the augmented values
        """
        keys = random.split(self.key, 1)
        return jax.numpy.vstack([random.normal(keys[0], (self.dim,1)), jax.numpy.array(0)])
        
    @partial(jit, static_argnums=(0,))
    def LSE(self, theta: jnp, X: jnp, y: jnp)-> jnp:
        """
        LSE in matrix form. We also use Jit por froze info at self to follow 
        the idea of functional programming on Jit for no side effects
        args:
            X: Data array at the GPU or CPU
            theta: Parameter w for weights and b for bias
            y: array of labels
        returns:
            the Loss function LSE under data X, labels y and theta initial estimation
        """
        return (jax.numpy.transpose(y - self.linear_model(X, theta))@(y - self.linear_model(X, theta)))[0,0]

    @partial(jit, static_argnums=(0,))
    def update(self, theta: jnp, X: jnp, y: jnp, lr):
        """
        Update makes use of the autograd at Jax to calculate the gradient descent.
        args:
            X: Data array at the GPU or CPU
            theta: Parameter w for weights and b for bias
            y: array of labels
            lr: Learning rate for Gradient Descent
        returns:
            the step update w(n+1) = w(n)-δ(t)𝜵L(w(n))        
        """
        return theta - lr * jax.grad(self.LSE)(theta, X, y)

        
    @partial(jit, static_argnums=(0,))
    def estimate_grsl(self, X, theta):
        """
        Estimation for the Gradient Descent version
        args:
            X: Data array at the GPU or CPU
            theta: Parameter w for weights and b for bias
        return:
            Estimation of data X under linear model
        """
        w = theta[:-1]
        b = theta[-1]
        return X@w+b
    
    def precision(self, y, y_hat):
        """
        Precision
        args:
            y: Real Labels
            y_hat: estimated labels
        return TP/(TP+FP)
        """
        TP=0
        FP=0
        for i in range(len(y)):
            if(y_hat[i]>0 and y[i]>0):
                TP+=1
            if(y_hat[i]>0 and y[i]<0):
                FP+=1

        #TP = sum(y_hat[y>0]>0)
        #FP = sum(y_hat[y>0]<0)
        precision_cpu = jax.jit(lambda x: x, device=self.cpus[0])(TP/(TP+FP))
        return float(precision_cpu)
    def accuracy(self, y, y_hat):
        """
        Precision
        args:
            y: Real Labels
            y_hat: estimated labels
        return TP/(TP+FP)
        """
        TP=0
        TN=0
        FP=0
        FN=0
        for i in range(len(y)):
            if(y_hat[i]>0 and y[i]>0) :
                TP+=1
            if(y_hat[i]<0 and y[i]<0):
                TN+=1
            if(y_hat[i]<0 and y[i]>0):
                FN+=1
            if(y_hat[i]>0 and y[i]<0):
                FP+=1

        #TP = sum(y_hat[y>0]>0)
        #FP = sum(y_hat[y>0]<0)
        accuracy_cpu = jax.jit(lambda x: x, device=self.cpus[0])((TP+TN)/(TP+FP+TN+FN))
        return float(accuracy_cpu)
    
    
    def gradient_descent(self, theta: jnp,  X: jnp, y: jnp, n_steps: int, lr = 0.001):
        """
        Gradient Descent Loop for the LSE Linear Model
        args:
            X: Data array at the GPU or CPU
            theta: Parameter w for weights and b for bias
            y: array of labels
            n_steps: number steps for the Gradient Loop
            lr: Learning rate for Gradient Descent   
        return:
            Updated Theta
        """
        for i in range(n_steps):
            theta = self.update(theta, X, y, lr)
            error=self.LSE( theta, X, y)
            print('este es el error en el paso i',i,error)
        return theta
    
    def model(self, theta, X, y, lr,n_steps,X_val,y_val,run_name):
        experiment_name = "lineal"
        experiment = mlflow.get_experiment_by_name(experiment_name)
        experiment_id = experiment.experiment_id
        with mlflow.start_run(experiment_id=experiment_id,run_name=run_name):
            mlflow.log_param('learning_rate',lr)
            mlflow.log_param('n_steps',n_steps)
            print('este es lr')
            theta=self.gradient_descent(theta,X,y,n_steps,lr)
            print('este es theta',theta)
            y_hat=self.estimate_grsl(X, theta)
            print('este es y_hat de training',y_hat)
            y_hat_val=self.estimate_grsl(X_val,theta)
            print('este es y_hat de val',y_hat_val)
            precision,accuracy=metrics.precision(y,y_hat),metrics.accuracy(y,y_hat)
            
            precision_val,accuracy_val=metrics.precision(y_val,y_hat_val),metrics.accuracy(y_val,y_hat_val)
            mlflow.log_metric('precision',precision)
            mlflow.log_metric('recall',accuracy)
            mlflow.log_metric('precision_val',precision_val)
            mlflow.log_metric('recall_val',accuracy_val)
            print('precision y recall',precision,accuracy)
        return y_hat,y_hat_val#,precision,recall,precision_val,recall_val

        
    ######################################################################################################
    #########vamos a hacer la implementación de la regularización de Ridge##################################

    def generate_canonicalRidge_estimator(self, X: jnp, y:jnp,la:jnp) -> jnp:
        """
        Cannonical LSE error solution for the Linearly separable classes 
        args:
            X: Data array at the GPU or CPU
            y: Label array at the GPU 
        returns:
            w: Weight array at the GPU or CPU
        """
        XX=jax.numpy.transpose(X)@X
        dimension=int(jnp.shape(XX)[0])
        I=jax.numpy.identity(dimension)
        return  jax.numpy.linalg.inv(XX+la*I)@jax.numpy.transpose(X)@y
    
    @staticmethod
    def estimate_cannonicalRidge(X: jnp, w: jnp)->jnp:
        """
        Estimation for the Gradient Descent version
        args:
            X: Data array at the GPU or CPU
            w: Parameter w under extended space
        return:
            Estimation of data X under cannonical solution
        """
        return X@w
    
    def model_ridge(self,X,y_est,l,incremento,samples,k_classes,clases,X_v,y_est_v,run_name):
        experiment_name = "Ridge"
        experiment = mlflow.get_experiment_by_name(experiment_name)
        experiment_id = experiment.experiment_id
        X_e = np.hstack([X, np.ones((samples,1))])
        samples2=X_v.shape[0]
        X_v= np.hstack([X_v, np.ones((samples,1))])
        l=0.02
        max_steps=2
        precision_list=[]
        accuracy_list=[]
        precision_list_v=[]
        accuracy_list_v=[]
        lbda=[]
        print('holaa')
        with mlflow.start_run(experiment_id= experiment_id, run_name=run_name) as run:
            mlflow.log_param('lambda',l)
            wR = self.generate_canonicalRidge_estimator(X_e, y_est,l)
            y_hatR = self.estimate_cannonicalRidge(X_e, wR)
            y_hatR_v = self.estimate_cannonicalRidge(X_e, wR)
            precision,accuracy=self.precision(y_est,y_hatR),self.accuracy(y_est,y_hatR)
            mlflow.log_metric('precision_list',precision)
            mlflow.log_metric('recall_list',accuracy)
            precision_v,accuracy_v=metrics.precision(y_est,y_hatR),metrics.accuracy(y_est_v,y_hatR_v)
            mlflow.log_metric('precision_val',precision_v)
            mlflow.log_metric('accuracy_val',accuracy_v)
            precision_list.append(precision)
            accuracy_list.append(accuracy)
            precision_list_v.append(precision_v)
            accuracy_list_v.append(accuracy_v)
            

            '''
            for i in range(max_steps):
                l=l+incremento
                lbda.append(l)
                mlflow.log_param('lambda',l)
                wR = self.generate_canonicalRidge_estimator(X_e, y_est,l)
                y_hatR = self.estimate_cannonicalRidge(X_e, wR)
                y_hatR_v = self.estimate_cannonicalRidge(X_e, wR)
                precision,accuracy=self.precision(y_est,y_hatR),self.accuracy(y_est,y_hatR)
                mlflow.log_metric('precision_list',precision)
                mlflow.log_metric('recall_list',accuracy)
                precision_v,accuracy_v=self.precision(y_est,y_hatR),self.accuracy(y_est_v,y_hatR_v)
                mlflow.log_metric('precision_val',precision_v)
                mlflow.log_metric('accuracy_val',accuracy_v)
                precision_list.append(precision)
                accuracy_list.append(accuracy)
                precision_list_v.append(precision_v)
                accuracy_list_v.append(accuracy_v)
            '''
            
            
            
            
            #grafica_pr=graficar.graficar_pr_ridge(accuracy_list,precision_list,accuracy_list_v,precision_list_v,lbda)
            #mlflow.log_figure(grafica_pr,'grafica_pr_ridge.png')
        return y_hatR,y_hatR_v,precision_list,