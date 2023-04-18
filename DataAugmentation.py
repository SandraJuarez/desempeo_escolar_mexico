import numpy as np
import random
from sklearn.neighbors import NearestNeighbors

def nearest_neighbour(x):
    nb=NearestNeighbors(n_neighbors=5,metric='euclidean',algorithm='kd_tree').fit(x)
    distancia,indices=nb.kneighbors(x)
    return indices

def Smote(x):
    rows=x.shape[0]
    indices=nearest_neighbour(x)
    new=[]
    for i in range (rows):
        mult=random.random()
        point_center=x[i,:]
        ind=np.random.randint(1,4)
        sx=int(indices[i,ind])
        second_point=x[sx,:]
        new_point=point_center+mult*(second_point-point_center)
        new.append(new_point)
    #new.append(x0)
    #clase=np.ones(rows)
    new=np.array(new)
    #new=np.concatenate((clase,new),axis=1)
    return new

if __name__=="__MAIN__":
    Smote()