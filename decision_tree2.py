import numpy as np
import pandas as pd
def get_gini(x,y,cut):
    size=x.shape[0]+0.001

    n1=np.count_nonzero(x<=cut)
    p10=np.count_nonzero((x<=cut) & (y==0))/(n1+0.001)
    #print('p10',p10)
    p11=np.count_nonzero((x<=cut)& (y==1))/(n1+0.001)
    #print('p11',p11)

    n2=np.count_nonzero(x>=cut)
    p00=np.count_nonzero((x>=cut) & (y==0))/(n2+0.001)
    #print('p00',p00)
    p01=np.count_nonzero((x>=cut)& (y==1))/(n2+0.001)
    #print('p01',p01)
    gini_impurity=(n1/size)*(1-p10**2-p11**2)+(n2/size)*(1-p00**2-p01**2)
    return gini_impurity


def get_best_feature(x,y):
    sizex=x.shape[1]
    best_cut_list=[]
    print('el shape',x.shape)
    for i in range(sizex):
        #print(i)
        #print('el shape',x[:,i])
        unicos=np.unique(x[:,i],return_counts=False)
        #print(unicos)
        sizeu=unicos.shape[0]
        gini_list=[]
        cut_list=[]
        feature=x[:,i]
        for j in range(1,sizeu):

            cut=unicos[j]
            #print('el cut',cut)
            cut_list.append(cut)
            
            
            gini_impurity=get_gini(feature,y,cut)
            
            gini_list.append(gini_impurity) #list of the ginies for a feature for each cut
        min=np.argmin(gini_list)
        best_cut=cut_list[min] #index of the best cut (from the list of possible cuts) for each feature
        best_cut_list.append(best_cut) #list of thebest cut of each feature
    best_feature=np.argmin(best_cut_list) #index of the best feature
    cut=best_cut_list[best_feature]
    print('el cut fue',cut)
    print('el best feature fue',best_feature)
    return best_feature,cut

def split_data(x,y):
    best_feature,cut=get_best_feature(x,y)
    index1=np.where(x[:,best_feature]<=cut)
    index2=np.where(x[:,best_feature]>cut)
    group1=x[index1[0],:]
    group2=x[index2[0],:]
    group1y=y[index1[0]]
    group2y=y[index2[0]]
    xg1=group1[:,best_feature]
    xg2=group2[:,best_feature]
    return group1,group2,group1y,group2y,xg1,xg2,cut

def train_tree(x,y,max_depth,gini_min_samples):
    count=0
    group11,group21,group11y,group21y,xg11,xg21,cut1=split_data(x,y)
    group12,group22,group12y,group22y,xg12,xg22,cut2=split_data(x,y)
    while count<=max_depth:
        print(count)
        group11,group21,group11y,group21y,xg11,xg21,cut1=split_data(group11,group11y)
        group12,group22,group12y,group22y,xg12,xg22,cut2=split_data(group12,group12y)

        samples11=group11.shape[0]
        samples12=group12.shape[0]

        samples21=group21.shape[0]
        samples22=group22.shape[0]

        gini11=get_gini(xg11,group11y,cut1)
        print('el gini 11',gini11)
        gini21=get_gini(xg21,group21y,cut1)
        print('el gini 21',gini21)
        gini12=get_gini(xg12,group12y,cut2)
        print('el gini 12',gini12)
        gini22=get_gini(xg22,group22y,cut2)
        print('el gini 22',gini22)
        if samples11<=gini_min_samples :
            print('eran muy pocas muestras',samples11)
            continue
        else:
            print('se continuo entrenando 11')
            train_tree(group11,group11y,max_depth,gini_min_samples)
            '''
        if samples21<=gini_min_samples:
            print('eran muy pocas muestras ',samples21)
            continue
        else:
            train_tree(group21,group21y,max_depth,gini_min_samples)  
            print('se continuo entrenando 21')
        
        if samples12<=gini_min_samples:
            print('eran muy pocas muestras',samples12)
            continue
        else:
            train_tree(group12,group12y,max_depth,gini_min_samples)
            print('se sigue entrenando 12')
        if samples22<=gini_min_samples:
            print('eran muy pocas muestras',samples22)
            continue
        else:
            train_tree(group22,group22y,max_depth,gini_min_samples) 
            print('se continuo entrenando 22')
            '''
        count=count+1

if __name__=="__MAIN__":
    train_tree()













