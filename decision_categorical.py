import numpy as np
import pandas as pd
from itertools import combinations
import csv
import jax.numpy as jnp
import jax
from jax import jit

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


def get_combinations(feature):
    unicos=np.unique(feature)
    cut=[]
    sizeu=unicos.shape[0]
    for j in range(1,sizeu):
            for subset in combinations(unicos,j):
                cut.append(subset)
    return cut


def get_best_feature(features_list,x,y):
    x=x[:,features_list]
    sizex=x.shape[1]
    best_cut_list=[]
    #print('el shape',x.shape)
    gini_features=[]
    
    for i in range(sizex):
        
        
        unicos=np.unique(x[:,i],return_counts=False)
        sizeu=unicos.shape[0]
        #print('el sizeu',sizeu)
        if unicos.shape[0] <=15:
            cut=[]
            gini_list=[]
            cut_list=[]
            feature=x[:,i]
            
                
            cut=get_combinations(feature)
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

def split_data(x,y,best_feature,cut):
    
    #print('el best feature es',best_feature)
    index1=list(np.unique(np.where(np.isin(x,cut))[0]))
    index2=list(np.unique(np.where(np.isin(x,cut,invert=True))[0]))
    group1=x[index1,:]
    group2=x[index2,:]

    #print(group1.shape,group2.shape)
    group1y=y[index1]
    group2y=y[index2]
    xg1=group1[:,best_feature]
    xg2=group2[:,best_feature]
    
    return group1,group2,group1y,group2y,xg1,xg2,cut,best_feature

def get_class_leaf(x,y):

    n=x.shape[0]
    #print('n1',x.shape,len(cut))
    p10=np.count_nonzero( y==0)/n
    #print('p10',p10)
    p11=np.count_nonzero(y==1)/n
    #print('p11',p11)
    prediction=np.argmax(np.array([p10,p11]))

    return prediction

def train_tree(x, y, gini_min,max_depth,current_depth=0,side='None',nodo_padre=0,nodo_hijo=0):
    
    if current_depth==0:
        features_list=list(range(x.shape[1]))
        np.savetxt('features_list.txt',features_list)
        np.savetxt('features_list_original.txt',features_list)
        np.savetxt('cut.csv',x)
        ######################para lo de los nodos
        df_node_data=pd.DataFrame(columns=['Nodo padre','Nodo hijo','Caracteristica','Valor','Gini impurity','div 1 No in class 1','div 1 No in class 2','div 2 No in class 1','div 2 No in class 2'])
        df_node_data.to_csv('node_data.csv', index=False)
    else:
        features_list=list(np.loadtxt('features_list.txt',ndmin=1))
        
        features_list=[int(x) for x in features_list]
    node_data=[]
    best_feature,cut=get_best_feature(features_list,x,y)
    #print('el cut',cut)
    
    #print('el x',x.shape)
    group1, group2, group1y, group2y, xg1, xg2, cut,feature = split_data(x,y,best_feature,cut)
    gini = get_gini(xg1, group1y, cut)
    #print('EL GINI ES',gini)
    #print('EL LEN DE CUT',len(cut))
    node_data.append({
    'Nodo padre':nodo_padre,
    'Nodo hijo':nodo_hijo, 
    'Caracteristica': best_feature, 
    'Valor': cut, 
    'Gini impurity': gini,
    'div 1 No in class 1': np.sum(group1y==0),
    'div 1 No in class 2': np.sum(group1y==1),
    'div 2 No in class 1': np.sum(group2y==0),
    'div 2 No in class 2': np.sum(group2y==1)
    })

    #añadimos los datos de cada nodo al dataframe
    df_node_data=pd.DataFrame(data=node_data, columns=['Nodo padre','Nodo hijo','Caracteristica','Valor','Gini impurity','div 1 No in class 1','div 1 No in class 2','div 2 No in class 1','div 2 No in class 2'], index=range(len(node_data)))    
    
    #guardar los datos de cada nodo en un archivo csv
    #df = pd.DataFrame(node_data)
    df_node_data.to_csv('node_data.csv', mode='a', header=False, index=False)

    with open('cut.csv', 'r', newline='') as file:
        reader = csv.reader(file)
        cut_old = list(reader)[0]
    if current_depth!=0:
        cut_old=tuple(cut_old)
        cut_old=np.array([float(x) for x in cut_old])
        cut_old=tuple(cut_old)
    #print(cut_old)
    if (len(cut)==len(cut_old)) and (cut_old==cut) :
        features_list.remove(features_list[best_feature])
        #print('el viejo corte era igual al nuevo')
        #print('se removió el feature',best_feature)
        #print('features list actualizada',features_list)
        np.savetxt('features_list.txt',features_list)
    if (len(cut)==1) :
        features_list.remove(features_list[best_feature])
        #print('se removió el feature',best_feature)
        np.savetxt('features_list.txt',features_list)
        #group1=group1[:,features_list]
        #print('ahora la shape de x',group1.shape)
    samples1 = group1.shape[0]
    #print('samples del grupo 1',samples1)
    #print('el cut es',cut)
    cut=list(cut)
    with open('cut.csv','w',newline='') as file:
        writer=csv.writer(file)
        writer.writerow(cut)
        
    
    
    lista=[['cut'],[cut],[feature],[features_list]]
    with open('arbol.csv','a',newline='') as file:
        writer=csv.writer(file)
        writer.writerow([elem for sublist in lista for elem in sublist ])
    
    if gini <= gini_min:
        #print('se llegó a un gini minimo')
        if side=='1':
            y_hat=get_class_leaf(group1,group1y)
            #print('se calculó la etiqueta de la PRIMERA hoja',y_hat)
            features_list=list(np.loadtxt('features_list_original.txt').astype(int))
            lista=[['hoja 1'],[y_hat]]
            with open('arbol.csv','a',newline='') as file:
                writer=csv.writer(file)
                writer.writerow([elem for sublist in lista for elem in sublist ])
        if side=='2':
            y_hat=get_class_leaf(group2,group2y)
            #print('se calculó la etiqueta de la SEGUNDA hoja',y_hat)
            features_list=list(np.loadtxt('features_list_original.txt').astype(int))
            lista=[['hoja 2'],[y_hat]]
            with open('arbol.csv','a',newline='') as file:
                writer=csv.writer(file)
                writer.writerow([elem for sublist in lista for elem in sublist ])
    
    elif max_depth==current_depth :
        #print('se llegó a la maxima profundidad')
       
        if side=='1':
            y_hat=get_class_leaf(group1,group1y)
            #print('se calculó la etiqueta de la PRIMERA hoja',y_hat)
            features_list=list(np.loadtxt('features_list_original.txt').astype(int))
            lista=[['hoja 1'],[y_hat]]
            with open('arbol.csv','a',newline='') as file:
                writer=csv.writer(file)
                writer.writerow([elem for sublist in lista for elem in sublist ])
        if side=='2':
            y_hat=get_class_leaf(group2,group2y)
            #print('se calculó la etiqueta de la SEGUNDA hoja',y_hat)
            features_list=list(np.loadtxt('features_list_original.txt').astype(int))
            max_depth=max_depth-1
            lista=[['hoja 2'],[y_hat]]
            with open('arbol.csv','a',newline='') as file:
                writer=csv.writer(file)
                writer.writerow([elem for sublist in lista for elem in sublist ])
    else:
        #print('seguimos entrenando ')
        #read csv para obtener el ultimo valor de nodo_hijo
        df_node_data=pd.read_csv('node_data.csv')
        new_nodo_hijo=df_node_data['Nodo hijo'].max()+1
        # Recursively call train_tree and capture the returned values
        train_tree(group1, group1y,gini_min, max_depth,current_depth+1,side='1',nodo_padre=nodo_hijo, nodo_hijo=new_nodo_hijo)
        
        #print('ahora estamos en la rama dos')
        #read csv para obtener el ultimo valor de nodo_hijo
        df_node_data=pd.read_csv('node_data.csv')
        new_nodo_hijo=df_node_data['Nodo hijo'].max()+1
        train_tree(group2, group2y,gini_min, max_depth,current_depth+1,side='2',nodo_padre=nodo_hijo, nodo_hijo=new_nodo_hijo)
    return node_data
#def make_predictions():
def find_next_node(df, X_to_predict , current_nodo_padre, current_nodo_hijo):
    #buscamos el nodo padre
    nodo_padre=df[df['Nodo hijo']==current_nodo_padre]['Nodo padre'].values[0]
    #print('nodo padre: ',nodo_padre)
    #buscamos la caracteristica
    caracteristica=df[df['Nodo hijo']==current_nodo_hijo]['Caracteristica'].values[0]
    #print('caracteristica: ',caracteristica)
    #buscamos el valor
    valor=df[df['Nodo hijo']==current_nodo_hijo]['Valor'].values[0]
    #print('valor: ',valor)
    valor=list(valor)
    #buscamos el hacia que nodo hijo se va a ir
    if np.isin(X_to_predict[caracteristica],valor):
            #print('nodo hijo derecho')
            #buscamos el nodo hijo con el valor mas bajo pero que tenga el nodo padre= lista_nodos[i]
            nodo_hijo=df[df['Nodo padre']==current_nodo_hijo]['Nodo hijo'].min()

            #previene de regresar en bucle a nodo raiz
            if nodo_hijo==0:
                nodo_hijo+=1

            #print('nodo hijo: ',nodo_hijo)
    else:
        #print('nodo hijo izquierdo')
        #buscamos el nodo hijo con el valor mas alto
        nodo_hijo=df[df['Nodo padre']==current_nodo_hijo]['Nodo hijo'].max()
        #print('nodo hijo: ',nodo_hijo)
    
    current_nodo_padre=nodo_padre
    current_nodo_hijo=nodo_hijo

    return nodo_padre,nodo_hijo

#------Metricas----------------
def precision_jax(y, y_hat):
    """
    precision
    args:
        y: Real Labels
        y_hat: estimated labels
    return TP/(TP+FP)
    """
    TP = jnp.sum((y > 0) & (y_hat > 0))
    FP = jnp.sum((y <= 0) & (y_hat > 0))

    #evitar division por cero
    precision_cpu = jax.lax.cond(
        TP + FP == 0,
        lambda _: 0.0,
        lambda _: TP / (TP + FP),
        operand=None,
    )
    

    return float(precision_cpu)


def recall_jax(y, y_hat):
    """
        recall
        args:
            y: Real Labels
            y_hat: estimated labels
        return TP/(TP+FN)
    """
    TP = jnp.sum((y > 0) & (y_hat > 0))
    FN = jnp.sum((y > 0) & (y_hat <= 0))

    #evitar division por cero
    recall_cpu = jax.lax.cond(
        TP + FN == 0,
        lambda _: 0.0,
        lambda _: TP / (TP + FN),
        operand=None,
    )

    return float(recall_cpu)

    
def accuracy_jax(y, y_hat):
    """
        accuracy
        args:
            y: Real Labels
            y_hat: estimated labels
        return  TP +TN/ TP +FP +FN+TN
    """
    TP = jnp.sum((y > 0) & (y_hat > 0))
    FP = jnp.sum((y <= 0) & (y_hat > 0))
    FN = jnp.sum((y > 0) & (y_hat <= 0))
    TN = jnp.sum((y <= 0) & (y_hat <= 0))
    
    #evitar division por cero         
    if (TP+FP+TN+FN)==0:
        return 0
    else:
        accuracy_cpu = jit(lambda x: x, device=jax.devices("cpu")[0])((TP+TN)/(TP+FP+TN+FN))
        return float(accuracy_cpu)                                              



    



def prediccion(X_to_predict, Y_labels=None):
    # Leer los datos del archivo CSV
    df = pd.read_csv('node_data.csv')
    #ordenamos los datos por el nodo padre pero tambien se ordenan los nodos hijos
    df=df.sort_values(by=['Nodo padre'])
    #print(df)
    #obtenemos el nodo raiz
    #nodo_raiz=df['Nodo padre'].min()
    #print('nodo raiz: ',nodo_raiz)
    #para cala row de X_to_predict
    prediccion=[]
    for row in range(len(X_to_predict)):
        row_to_predict=X_to_predict[row]
        #print('X_to_predict: ',row_to_predict.shape)
        #inicializamos para el primer nodo raiz
        current_nodo_padre=0
        current_nodo_hijo=0
        while True:
            #si solo tenemos un nodo osea un stump entonces solo se hace la prediccion y no se busca el siguiente nodo
            if len(df['Nodo padre'].unique())!=1:
                current_nodo_padre,current_nodo_hijo=find_next_node(df, row_to_predict , current_nodo_padre, current_nodo_hijo)
                
            
            #si el current hijo no esta en la lista de nodos hijos entonces es una hoja
            if current_nodo_hijo not in df['Nodo padre'].values or len(df['Nodo padre'].unique())==1:
                #vemos caracteristica y valor
                caracteristica=df[df['Nodo hijo']==current_nodo_hijo]['Caracteristica'].values[0]
                print('caracteristica: ',caracteristica)
                valor=df[df['Nodo hijo']==current_nodo_hijo]['Valor'].values[0]
                print('valor: ',valor)
                #calido si es menor o mayor
                if np.isin(row_to_predict[caracteristica],valor):
                    #vemos que clase es la que tiene mas cantidad de datos en df['div 1No in class 1'] y df['div 1 No in class 2'] en ese nodo
                    if df[df['Nodo hijo']==current_nodo_hijo]['div 1 No in class 1'].values[0]>df[df['Nodo hijo']==current_nodo_hijo]['div 1 No in class 2'].values[0]:
                        print('prediccion: 1')
                        prediccion.append(0)
                    else:
                        print('prediccion: 2')
                        prediccion.append(1)
                    break
                else:
                    #vemos que clase es la que tiene mas cantidad de datos en df['div 2 No in class 1'] y df['div 2 No in class 2'] en ese nodo
                    if df[df['Nodo hijo']==current_nodo_hijo]['div 2 No in class 1'].values[0]>df[df['Nodo hijo']==current_nodo_hijo]['div 2 No in class 2'].values[0]:
                        #print('prediccion: 1')
                        prediccion.append(0)
                    else:
                        #print('prediccion: 2')
                        prediccion.append(1)
                    break
    
    #if exist  y_labels
    if Y_labels is not None:
        #metrics
        #convert to jnp
        Y_labels=jnp.array(Y_labels)
        prediccion=jnp.array(prediccion)
        #print('Y_labels: ',Y_labels)
        #print('prediccion: ',prediccion)

        precision=precision_jax(Y_labels, prediccion)
        recall=recall_jax(Y_labels, prediccion)
        accuracy=accuracy_jax(Y_labels, prediccion)
        return prediccion, precision, recall, accuracy
    else:
        return prediccion
    
    
    

    

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
    prediccion()













