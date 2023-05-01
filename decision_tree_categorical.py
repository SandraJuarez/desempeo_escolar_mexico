import numpy as np
import pandas as pd
def get_gini(x,y,cut):
    size=x.shape[0]
    
    n1=np.count_nonzero(x==cut)
    p10=np.count_nonzero((x==cut) & (y==0))/(n1)
    #print('p10',p10)
    p11=np.count_nonzero((x==cut)& (y==1))/(n1)
    #print('p11',p11)

    n2=np.count_nonzero(x!=cut)
    p00=np.count_nonzero((x!=cut) & (y==0))/(n2)
    #print('p00',p00)
    p01=np.count_nonzero((x!=cut)& (y==1))/(n2)
    #print('p01',p01)
    gini_impurity=(n1/size)*(1-p10**2-p11**2)+(n2/size)*(1-p00**2-p01**2)
    return gini_impurity


def get_best_feature(x,y):
    sizex=x.shape[1]
    best_cut_list=[]
    #print('el shape',x.shape)
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
    print(group1.shape)
    group2=x[index2[0],:]
    group1y=y[index1[0]]
    group2y=y[index2[0]]
    xg1=group1[:,best_feature]
    xg2=group2[:,best_feature]
    return group1,group2,group1y,group2y,xg1,xg2,cut
def train_tree(x, y, gini_min,max_depth,current_depth=0):
    
    group1, group2, group1y, group2y, xg1, xg2, cut = split_data(x,y)
    samples1 = group1.shape[0]
    print('en la primera iteracion tenemos las siguientes samples',samples1)

    gini = get_gini(xg1, group1y, cut)
    print('EL GINI ES',gini)
    

    #if samples1 <= gini_min_samples:
     #   print('eran muy pocas muestras', samples1)
    if gini <= gini_min:
        print('gini minimo')
        
    elif max_depth==current_depth :
        print('se llegó a la maxima profundidad')
       
    else:
        print('se continuo entrenando ')
        
        # Recursively call train_tree and capture the returned values
        train_tree(group1, group1y,gini_min, max_depth,current_depth+1)
        print('ahora en la rama dos')
        train_tree(group2, group2y,gini_min, max_depth,current_depth+1)
    
    

    

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
