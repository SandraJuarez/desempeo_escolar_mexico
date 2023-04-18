import numpy as np
import pandas as pd
import random
import itertools
from itertools import combinations
import random


#first,we bring our data from the previously cleaned dataset and we separate in classes
def separate(X_train,y_train):
    ''' 
   input -> X_train, y_train: numpy array
    return -> separated data into two class. It works only for binary problems
    '''
    df_x=pd.DataFrame(X_train) 
    df_y=pd.DataFrame(y_train, columns=['y']) 
    df=pd.concat([df_x, df_y], axis=1)   
    grupos=df.groupby(['y'])
    df1=pd.DataFrame(grupos.get_group(1))
    df2=pd.DataFrame(grupos.get_group(0)) 
    df1=df1.drop(df1.columns[-1],axis=1)
    df2=df2.drop(df2.columns[-1],axis=1)
    #reconvert to numpy array 
    X_class_1=df1.to_numpy() 
    X_class_2=df2.to_numpy() 
    return X_class_1,X_class_2


#if we have too many features, it will be to much for the sequential backwards selection
#then, we need to use the following function to separate the data in random groups and select the best
#the best group is selected in base to the maximum norm of the mean
def group_mus(X_class_1,X_class_2,num_features,num_grupos,size_grupos):
    ''' 
    This function takes the total of features and selects a random group of features with the maximum norm between them
    args:
        X_class1: Data array of n samples x m features belonging to class 1
        X_class2: Data array of n samples x m features belonging to class 1
        num_features: integer, number of features
        num_grupos: number of groups in which we want to divide the total of features
        size_grupos: number of features in each group
    
    return:
        x1_ganadores: array of size (samples, winning features)
        x2_ganadores: array of size (samples, winning features)
        ganadores: indexes, (taken from the original array) of the winning features

    '''
    lista_index=random.sample(list(range(0,num_features)),num_features) #desordenamos los números del 0 al num_features
    list_mu=[]

    mu1=np.zeros(size_grupos) #in this vector we'll save the components of the average for each feature in class 1
    mu2=np.zeros(size_grupos) #in this vector we'll save the components of the average for each feature in class 2
    r=0
    for i in range(num_grupos):
        
        j=lista_index[i+r] #the column number is taken from our desorderd list 'lista_index'
        for k in range(size_grupos):
            if j==103:
             j=100
            mu1[k]=np.mean(X_class_1[:,j]) 
            mu2[k]=np.mean(X_class_2[:,j])
        mu_resta=np.linalg.norm(mu1-mu2) #we take the norm of the substraction of averages in each class
        list_mu.append(mu_resta)
        
        
        r=r+size_grupos #for i=0, we take the first num_grupos from 'lista_index', for i=1 the next group, etc
    #print('lista mus',list_mu)
    ganador=np.argmax(mu_resta) #we take the index of the greatest separation between mu1 and mu2
    x1_ganadores=np.zeros((X_class_1.shape[0],size_grupos))
    x2_ganadores=np.zeros((X_class_2.shape[0],size_grupos))
    ganadores=[]
    for i in range(size_grupos):
        index=lista_index[size_grupos*ganador+i]
        if index==103:
            index=100
        x1_ganadores[:,i]=X_class_1[:,index]
        x2_ganadores[:,i]=X_class_2[:,index]
        ganadores.append(index)

    return x1_ganadores,x2_ganadores,ganadores
##############################################################################
##########################################################################
####Now we create the function for the SEQUENTIAL BACKWARDS SELECTION#######

  

def sequential_backwards(size,x1,x2,min_group):
    ''' 
    This functions performs a sequential backwards selection

    args:
        size: integer, size of the original input data
        x1: array of data belonging to class 1
        x2: array of data belonging to class 2
        min_group: integer, size of group we want to achieve
    
    return: 
        x1
        x2
        ganadores
        col

    '''

    lista=[]
    for i in range(size):
        lista.append(i)
    for i in reversed(range(min_group,size+1)):
        total=i #mumero de grupos que tenemos
        size_grupos=int(i-1) #miembros de cada grupo
        min_group=min_group+1
        
        index_combinacion=list(combinations(lista,size_grupos)) #lista de tamaño 'total' de sublistas de tamaño 'size_grupos'
        mus=[]
        for j in range(total):
            iter=index_combinacion[j] #iter es una lista de tamaño size_grupos, se trae la sublista j de index_combinations           
            mu1=np.zeros(size_grupos) 
            mu2=np.zeros(size_grupos)
            mu_r=np.zeros(size_grupos)
            for k in range(size_grupos):
                mu1[k]=np.mean(x1[:,iter[k]])
                mu2[k]=np.mean(x2[:,iter[k]])
                #print('este es mu1 y mu2',mu1[k],mu2[k],k)
                mu_r[k]=mu1[k]-mu2[k]           
            mu_norm=np.linalg.norm(np.abs(mu_r))
            mus.append(mu_norm)     
        ganador=np.argmax(mus)
        lista=index_combinacion[ganador] 
        ganadores=[]
        x1g=np.zeros((x1.shape[0],size_grupos+1))
        x2g=np.zeros((x2.shape[0],size_grupos+1))
        for i in range(size_grupos-1):
            index=lista[i]#***
            ganadores.append(index)
            x1g[:,i]=x1[:,index]
            x2g[:,i]=x2[:,index]
    print('Los índices ganadores fueron:',index_combinacion[ganador])
    return x1g,x2g,ganadores


if __name__=="__MAIN__":
    separate()
    group_mus()
    sequential_backwards()