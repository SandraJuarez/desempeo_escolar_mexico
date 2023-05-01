import jax
import jax_metrics as jm
import jax.numpy as jnp
from jax import grad, jit, vmap
from functools import partial
from jax import random
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy
import metrics
import mlflow

# Switch off the cache 
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'

class Logistic_Regression():
    """
    Basic Model + Quasi Newton Methods
    """
    def __init__(self, regularization='l2', method_opt='classic_model'):
        self.regularization = regularization
        self.method_opt = method_opt
        self.error_gradient = 0.001
        self.key = random.PRNGKey(0)
        # You need to add some variables
        self.W = None

    @staticmethod
    @jit
    def logistic_exp(W:jnp, X:jnp)->jnp:
        """
        Generate all the w^T@x values 
        args:
            W is a k-1 x d + 1
            X is a d x N
        """
        return jnp.exp(W@X)

    @staticmethod
    @jit
    def logistic_sum(exTerms: jnp)->jnp:        
        """
        Generate all the w^T@x values 
        args:
            W is a k-1 x d 
            X is a d x N
        """
        temp = jnp.sum(exTerms, axis=0)
        n = temp.shape[0]
        return jnp.reshape(1.0+temp, newshape=(1, n))

    @staticmethod
    @jit
    def logit_matrix(Terms: jnp, sum_terms: jnp)->jnp:
        """
        Generate matrix
        """
        divisor = 1/sum_terms
        n, _ = Terms.shape
        replicate = jnp.repeat(divisor, repeats=n, axis=0 )
        logits = Terms*replicate
        return jnp.vstack([logits, divisor])
    
    @partial(jit, static_argnums=(0,))
    
    def model(self, W:jnp, X:jnp, Y_hot:jnp,lamda)->jnp:
        """
        Logistic Model, and regularized model with lamda !=0
        """
        W = jnp.reshape(W, self.sh)
        terms = self.logistic_exp(W, X)
        sum_terms = self.logistic_sum(terms)
        matrix = self.logit_matrix(terms, sum_terms)
        return jnp.sum(jnp.sum(jnp.log(matrix)*Y_hot, axis=0), axis=0) + lamda*jnp.trace(jnp.transpose(W)@(W))#devuelve el error total de la suma de las probabilidades y*log(x|w)
    

        #
        #def model(self, W:jnp, X:jnp, Y_hot:jnp,lamda=0)->jnp:
        #W = jnp.reshape(W, self.sh)
        #print(W.shape,X.shape,Y_hot.shape)
        #Z = - W @ X
        #N = X.shape[0]
        #loss = 1/N * (jnp.trace(W @ X @ jnp.transpose(Y_hot)) + jnp.sum(jnp.log(jnp.sum(jnp.exp(Z), axis=0))))
        #print('loss', loss.shape)
        #return loss
        
    




    @staticmethod
    def one_hot(Y: jnp):
        """
        One_hot matrix
        """
        numclasses = len(jnp.unique(Y))
        return jnp.transpose(jax.nn.one_hot(Y, num_classes=numclasses))
    
    def generate_w(self, k_classes:int, dim:int)->jnp:
        """
        Use the random generator at Jax to generate a random generator to instanciate
        the augmented values
        """
        key = random.PRNGKey(0)
        keys = random.split(key, 1)
        return jnp.array(random.normal(keys[0], (k_classes, dim)))

    @staticmethod
    def augment_x(X: jnp)->jnp:
        """
        Augmenting samples of a dim x N matrix
        """
        N = X.shape[1]
        return jnp.vstack([X, jnp.ones((1, N))])
     
   
    def fit(self, X: jnp, Y:jnp,alpha,tol,lamda)->None:
        """
        The fit process
        """
        nclasses = len(jnp.unique(Y))
        X = self.augment_x(X)
        dim = X.shape[0]
        W = self.generate_w(nclasses-1, dim)
        Y_hot = self.one_hot(Y)
        print(lamda)
        self.W = getattr(self, self.method_opt, lambda W, X, Y_hot: self.error() )(W, X, Y_hot,Y,alpha,tol,lamda)
        return self.W
    
    @staticmethod
    def error()->None:
        """
        Only Print Error
        """
        raise Exception("Opt Method does not exist")
    def estimate_prob(self, X:jnp)->jnp:
        """
        Estimate Probability
        """
        X = self.augment_x(X)
        terms = self.logistic_exp(self.W, X)
        sum_terms = self.logistic_sum(terms)
        matrix = self.logit_matrix(terms, sum_terms)    #shape (classes, samples)
        return matrix
    
    def estimate(self, X:jnp)->jnp:
        """
        Estimation
        """
        X = self.augment_x(X)
        terms = self.logistic_exp(self.W, X)
        sum_terms = self.logistic_sum(terms)
        matrix = self.logit_matrix(terms, sum_terms)
        return jnp.argmax(matrix, axis=0)
    
    def precision(self, y, y_hat,y_val,y_hat_val,learning_rate,lamda,run_name):
        """
        Precision
        args:
            y: Real Labels
            y_hat: estimated labels
        return TP/(TP+FP)
        """
        TP = sum(y_hat == y)
        FP = sum(y_hat != y)
        TPv = sum(y_hat_val == y_val)
        FPv = sum(y_hat_val != y_val)
        experiment_name = "logistic"
        experiment = mlflow.get_experiment_by_name(experiment_name)
        experiment_id = experiment.experiment_id
        with mlflow.start_run(experiment_id= experiment_id, run_name=run_name) as run:
            mlflow.log_param('learning_rate',learning_rate)
            mlflow.log_param('lamda',lamda)
            precision=(TP/(TP+FP))
            precisionv=(TPv/(TPv+FPv))
            accuracy=metrics.accuracy(y,y_hat)
            recall=metrics.recall(y,y_hat)
            precision=metrics.precision(y,y_hat)
            mlflow.log_metric('accuracy',accuracy)
            mlflow.log_metric('precision',precision)
            mlflow.log_metric('recall',recall)
            mlflow.log_metric('recall',accuracy)

            accuracyv=metrics.accuracy(y_val,y_hat_val)
            recallv=metrics.recall(y_val,y_hat_val)
            precisionv=metrics.precision(y_val,y_hat_val)
            print('EL ACCURACY de val ES',accuracyv)
            mlflow.log_metric('accuracy validation',accuracyv)
            mlflow.log_metric('precision validation',precisionv)
            mlflow.log_metric('recall validation',recallv)
            mlflow.log_metric('recall validation',accuracyv)
        return (TP/(TP+FP)).tolist()
    
    def classic_model(self, W:jnp, X:jnp, Y_hot:jnp, Y:jnp,alpha:float, tol, lamda)->jnp:
        """
        The naive version of the logistic regression
        """
        print('lambda',lamda)
        n, m = W.shape 
        self.sh = (n, m)
        
        Grad = jax.grad(self.model, argnums=0)(jnp.ravel(W), X, Y_hot,lamda)
        
        loss = self.model(jnp.ravel(W), X, Y_hot,lamda=0)
        cnt = 0
        max_step=int(100)
        while True:
            Hessian = jax.hessian(self.model, argnums=0)(jnp.ravel(W), X, Y_hot,lamda)
            W = W - alpha*jnp.reshape((jnp.linalg.inv(Hessian)@Grad), self.sh)
            Grad =  jax.grad(self.model, argnums=0)(jnp.ravel(W), X, Y_hot,lamda)
            old_loss = loss
            loss = self.model(jnp.ravel(W), X, Y_hot,lamda)
            if cnt%30 == 0:
                print(f'{self.model(jnp.ravel(W), X, Y_hot,lamda)}')
                
            if  jnp.abs(old_loss - loss) < tol:
                print('la tolerancia fue menor')
                break
            if cnt>max_step:
                break
            cnt +=1
        #y_hat=self.estimate(X)
        #precision=self.precision(Y,y_hat)
        #print('la precision es',precision)
        return W




    



    
    