import numpy as np
import pandas as pd
from itertools import combinations
import csv

def get_gini(x,y,cut):
    size=x.shape[0]

    #n1=np.count_nonzero(x==cut)
    n1=np.count_nonzero(np.isin(x,cut))
    #print('n1',x.shape,len(cut))
    p10=np.count_nonzero(np.isin(x,cut) & (y==0))/(n1)
    #print('p10',p10)
    p11=np.count_nonzero(np.isin(x,cut)& (y==1))/(n1)
    #print('p11',p11)

    n2=np.count_nonzero(np.isin(x,cut,invert=True))
    p00=np.count_nonzero(np.isin(x,cut,invert=True) & (y==0))/(n2)
    #print('p00',p00)
    p01=np.count_nonzero(np.isin(x,cut,invert=True)& (y==1))/(n2)
    #print('p01',p01)
    gini_impurity=(n1/size)*(1-p10**2-p11**2)+(n2/size)*(1-p00**2-p01**2)
    return gini_impurity


def get_best_feature(x,y):
    sizex=x.shape[1]
    best_cut_list=[]
    #print('el shape',x.shape)
    gini_features=[]
    for i in range(sizex):
        #print(i)
        #print('el shape',x[:,i])
        unicos=np.unique(x[:,i],return_counts=False)

        sizeu=unicos.shape[0]
        #print('el sizeu',sizeu)
        if unicos.shape[0] <=15:
            cut=[]
            gini_list=[]
            cut_list=[]
            feature=x[:,i]
            for j in range(1,sizeu):
                for subset in combinations(unicos,j):
                    cut.append(subset)
                
        
            #print(len(cut))    
            for j in range(1,len(cut)):
                gini_impurity=get_gini(feature,y,cut[j]) #we take a list of the lists of cut
                gini_list.append(gini_impurity) #list of the ginies for a feature for each cut
            min=np.argmin(gini_list)
            gini_features.append(gini_list[min])
            best_cut=cut[min] #index of the best cut (from the list of possible cuts) for each feature
            best_cut_list.append(best_cut) #list of thebest cut of each feature
    best_feature=np.argmin(gini_features) #index of the best feature
    cut=best_cut_list[best_feature]        
    
    return best_feature,cut

def split_data(x,y):
    best_feature,cut=get_best_feature(x,y)
    print('el best feature es',best_feature)
    index1=np.unique(np.where(np.isin(x,cut))[0])
    #print('los indices',index1.shape)
    index2=np.unique(np.where(np.isin(x,cut,invert=True))[0])
    group1=x[index1,:]
    group2=x[index2,:]

    print(group1.shape,group2.shape)
    group1y=y[index1]
    group2y=y[index2]
    xg1=group1[:,best_feature]
    xg2=group2[:,best_feature]
    return group1,group2,group1y,group2y,xg1,xg2,cut,best_feature

def get_class_leaf(x,y):

    n=x.shape[0]
    #print('n1',x.shape,len(cut))
    p10=np.count_nonzero( y==0)/n
    print('p10',p10)
    p11=np.count_nonzero(y==1)/n
    print('p11',p11)
    prediction=np.argmax(np.array([p10,p11]))

    return prediction

def train_tree(x, y, gini_min,max_depth,current_depth=0,side='None'):
    
    group1, group2, group1y, group2y, xg1, xg2, cut,feature = split_data(x,y)
    samples1 = group1.shape[0]
    print('samples del grupo 1',samples1)
    print('el cut es',cut)
    

    gini = get_gini(xg1, group1y, cut)
    print('EL GINI ES',gini)
    
    lista=[['cut'],[cut],[feature]]
    with open('arbol.csv','a',newline='') as file:
        writer=csv.writer(file)
        writer.writerow([elem for sublist in lista for elem in sublist ])
    
    if gini <= gini_min:
        print('se llegó a un gini minimo')
        if side=='1':
            y_hat=get_class_leaf(group1,group1y)
            print('se calculó la etiqueta de la PRIMERA hoja',y_hat)
            lista=[['hoja 1'],[y_hat]]
            with open('arbol.csv','a',newline='') as file:
                writer=csv.writer(file)
                writer.writerow([elem for sublist in lista for elem in sublist ])
        if side=='2':
            y_hat=get_class_leaf(group2,group2y)
            print('se calculó la etiqueta de la SEGUNDA hoja',y_hat)
            lista=[['hoja 2'],[y_hat]]
            with open('arbol.csv','a',newline='') as file:
                writer=csv.writer(file)
                
                writer.writerow([elem for sublist in lista for elem in sublist ])

        
    elif max_depth==current_depth :
        print('se llegó a la maxima profundidad')
       
        if side=='1':
            y_hat=get_class_leaf(group1,group1y)
            print('se calculó la etiqueta de la PRIMERA hoja',y_hat)
            lista=[['hoja 1'],[y_hat]]
            with open('arbol.csv','a',newline='') as file:
                writer=csv.writer(file)
                writer.writerow([elem for sublist in lista for elem in sublist ])
        if side=='2':
            y_hat=get_class_leaf(group2,group2y)
            print('se calculó la etiqueta de la SEGUNDA hoja',y_hat)
            max_depth=max_depth-1
            lista=[['hoja 2'],[y_hat]]
            with open('arbol.csv','a',newline='') as file:
                writer=csv.writer(file)
                writer.writerow([elem for sublist in lista for elem in sublist ])
    else:
        print('seguimos entrenando ')
        
        # Recursively call train_tree and capture the returned values
        train_tree(group1, group1y,gini_min, max_depth,current_depth+1,side='1')
        print('ahora estamos en la rama dos')
        train_tree(group2, group2y,gini_min, max_depth,current_depth+1,side='2')

#def make_predictions():


    
    
    

    

'''
def train_tree(x,y,max_depth,gini_min_samples,gini_min):
    count=0
    
    group1,group2,group1y,group2y,xg1,xg2,cut=split_data(x,y)
    samples1=group1.shape[0]
    #group12,group22,group12y,group22y,xg12,xg22,cut2=split_data(x,y)
    while samples1>=gini_min_samples:
        
        group1,group2,group1y,group2y,xg1,xg2,cut=split_data(group1,group1y)
        #group12,group22,group12y,group22y,xg12,xg22,cut2=split_data(group12,group12y)

        samples1=group1.shape[0]
        print(samples1)

        gini=get_gini(xg1,group1y,cut)
        print(gini)
        
        if samples1<=gini_min_samples :
            print('eran muy pocas muestras',samples1)
            continue
        #if gini<=gini_min:
         #   print('se llegó al gini mínimo')
          #  continue
        else:
            print('se continuo entrenando ')

            group1_new,group1y_new=train_tree(group1,group1y,max_depth,gini_min_samples,gini_min)
            print('ahora entrenamos la otra rama')
            group2_new,group2y_new=train_tree(group2,group2y,max_depth,gini_min_samples,gini_min)
            group1=group1_new
            group1y=group1y_new
        return group1,group1y
          
 '''       

if __name__=="__MAIN__":
    train_tree()













