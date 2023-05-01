
import os
import numpy as np
import pandas as pd
import metrics
import math as mt
import mlflow
import importlib
import mlflow
importlib.reload(metrics)


from matplotlib import pyplot 
# Switch off the cache 
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'
def get_gaus(features,samples,sigma,mu,x):
    ''' 
    Necesitamos una Gaussiana para cada clase
    Esta gausiana contiene la matriz Xk con r features
    Es necesario llamar la función para cada Xk perteneciente a la clase k
    '''
    gausiana=np.zeros(samples)
    sigma=sigma+0.001*np.identity(features)
    det_s=np.linalg.det(sigma)
    inv_s=np.linalg.inv(sigma)

    for i in range(samples):
        gausiana[i]=np.exp(-0.5*np.transpose(x[i,:]-mu)@inv_s@(x[i,:]-mu))*1/(np.sqrt((2*mt.pi)**features*det_s))
    return gausiana

def p_get_init(clases,label,train_labels):
    prob=np.zero(clases)
    contador=0
    for k in range(clases):
        if train_labels==label:
            contador=+1
        prob[k]=contador/len(train_labels)
        contador=0
    return  prob


def denominador_gama(p,features,samples,sigma1,sigma2,mu1,mu2,x1,x2):
    ''' 
    sum_k(pi_k*N_k) es la misma para todos, sumamos sobre todas las clases
    '''
    x=np.vstack((x1,x2))
    gausiana1=get_gaus(features,samples,sigma1,mu1,x)
    gausiana2=get_gaus(features,samples,sigma2,mu2,x)
    
    denominador=p[0]*gausiana1+p[1]*gausiana2
    
    return denominador
def get_gama(features,samples,p,gausiana,denominador):
    ''' 
    Necesitamos una gama para cada clase
    Gamma va a ser una matriz
    '''
    gama=np.zeros(samples)
    
    for i in range(samples):
        gama[i]=p*gausiana[i]/denominador[i]


    return gama

def get_nk(gama):
    ''' 
    Nk debe ser calculado para cada clase
    '''
    Nk=np.sum(gama)    
    return Nk

def get_mu(features,gama,x,Nk):
    ''' 
    Vamos a obtener el vector de mus para cada clase
    '''
    mu=np.zeros(features)
    for j in range(features):
        mu[j]=np.sum(gama*x[:,j])/Nk
    return mu

def cov(x0,y0,mu1,mu2,gama,Nk):
    #covr=np.sum(gama*np.transpose(x0-mu[0])@(y0-mu[1]))/Nk
    covr=np.sum(gama*(x0-mu1)@(y0-mu2))/Nk
    return covr
def get_sigma(samples,features,gama,x,Nk,mu):
    ''' 
    Obtenemos la sigma para cada clase
    Sigma es una matriz de features*features
    '''
    #X=np.zeros(samples)
    sigma=np.zeros((features,features))
    
    
    for i in range(samples):
        
        x[i,:]=np.reshape(x[i,:],(1,features))
        mu=np.reshape(mu,(1,features))
        sigma_i=(1/Nk)*gama[i]*np.transpose(x[i,:]-mu)@(x[i,:]-mu)
        sigma=sigma+sigma_i
    

    #for jy in range(features):
        #for jx in range(features):
            #sigma[jx,jy]=np.sum(gama*(x[:,jy]-mu[jy])@np.transpose(x[:,jx]-mu[jx]))/Nk
    #sigma=np.array([[cov(xx, xx,mu[0],mu[0],gama,Nk), cov(xx, xy,mu[0],mu[1],gama,Nk)], \
     #               [cov(xy, xx,mu[1],mu[0],gama,Nk), cov(xy, xy,mu[1],mu[1],gama,Nk)]])
    
    
    return sigma                    
def get_pi(clases,sample1,sample2,Nk1,Nk2):
    samples=sample1+sample2
    p=np.zeros(clases)    
    p[0]=Nk1/samples
    p[1]=Nk2/samples
    return p

def solve(clases,features,samples1,samples2,p,mu1,mu2,sigma1,sigma2,x1,x2):
    #p_ge_init(clases,label,train_labels)
    
    
    contador=0
    #while any(ep)>any(np.array([0.001,0.001] )):
    max_step=20
    for i in range(max_step):
        contador+=1
        print(contador)
        old1=mu1
        old2=mu2

        gausiana1=get_gaus(features,samples1,sigma1,mu1,x1)
        gausiana2=get_gaus(features,samples2,sigma2,mu2,x2)
        print('esta es la gausiana',gausiana1[:10],gausiana2[:10])
        samples=samples1+samples2
        denominador=denominador_gama(p,features,samples,sigma1,sigma2,mu1,mu2,x1,x2)
        print('este es el denominador',denominador[:10])
        p1=p[0]
        p2=p[1]
        

        gama1=get_gama(features,samples1,p1,gausiana1,denominador[:samples1])
        gama2=get_gama(features,samples2,p2,gausiana2,denominador[samples1:])
        
        Nk1=get_nk(gama1)
        Nk2=get_nk(gama2)
        print('estos son los nk', Nk1,Nk2)

        mu1=get_mu(features,gama1,x1,Nk1)
        mu2=get_mu(features,gama2,x2,Nk2)

        sigma1=get_sigma(samples1,features,gama1,x1,Nk1,mu1)
        sigma2=get_sigma(samples2,features,gama2,x2,Nk2,mu2)
        
        samples=samples1+samples2
        p=get_pi(clases,samples1,samples2,Nk1,Nk2)
        ep1=np.linalg.norm(mu1-old1)
        ep2=np.linalg.norm(mu2-old2)
        print(mu1,sigma1,mu2,sigma2,p)
        if ep1<0.001 and ep2<0.001:
            break
        
    return mu1,sigma1,mu2,sigma2,p,Nk1,Nk2

def get_prediction(x,mu1,sigma1,mu2,sigma2,features,samples,y,ep,run_name):
    ''' 
    important:  mu1 has to be the mean of the class labeled as '0' and mu2 the mean of the labeled as '1'
    '''
    experiment_name = "Mixture"
    experiment = mlflow.get_experiment_by_name(experiment_name)
    experiment_id = experiment.experiment_id
    with mlflow.start_run(experiment_id=experiment_id,run_name=run_name):
        mlflow.log_param('Features',features)
        mlflow.log_param('Stop criteria',ep)
        p1=get_gaus(features,samples,sigma1,mu1,x)
        p2=get_gaus(features,samples,sigma2,mu2,x)
        stak=np.vstack((p1,p2))
        y_hat=np.argmax(stak,axis=0)
        #TP = sum(y_hat == y)
        #FP = sum(y_hat != y)
        precision=metrics.precision(y,y_hat)
        accuracy=metrics.accuracy(y,y_hat)
        recall=metrics.recall(y,y_hat)
        #precision=(TP/(TP+FP))
        mlflow.log_metric('accuracy',accuracy)
        mlflow.log_metric('precision',precision)
        mlflow.log_metric('recall',recall)
        mlflow.log_metric('recall',accuracy)
        print('La precision, accuracy y recall',precision,accuracy,recall)
    return y_hat,precision
if __name__=="__MAIN__":
    solve()
    get_prediction()