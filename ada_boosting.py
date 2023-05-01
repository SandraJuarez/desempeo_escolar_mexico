import numpy as np
import linear
from linear import Linear_Model
import jax.numpy as jnp
import decision_categorical as dt
import pandas as pd
import ast
import metrics
import mlflow
from unittest.mock import MagicMock
import mlp_mlflow
from mlp_mlflow import MultilayerPerceptron
import y_hot as hot

def get_prediction(x_train,y_train,x_v,y_v):
    dim=int(x_train.shape[1])
    y_train=y_train
    y_vl=y_v

    for i in range(y_train.shape[0]):
        if y_train[i]==0.0:
            y_train[i]=-1
    for i in range(y_v.shape[0]):
        if y_v[i]==0:
            y_vl[i]=-1

    x_train=jnp.array(x_train)
    y_train=jnp.array(y_train)
    x_vl=jnp.array(x_v)
    y_vl=jnp.array(y_vl)
    labels=jnp.array([-1,1])
    k_classes=2
    n_steps=100
    lr=1e-6
    samples=x_train.shape[0]
    #linear model
    #importlib.reload(linear)
    modelo=Linear_Model(dim)
    theta = modelo.generate_theta() #inicializamos theta con valores aleatorios
    
    y_train=jnp.reshape(y_train,(y_train.shape[0],1))
    
    y_hat=modelo.model_boost(theta, x_train, y_train, lr,n_steps,x_vl,y_vl)[0]
    print('EL Y_HAT SIN RAVEL',y_hat)
    y_hat=np.sign(y_hat)
    print('EL Y_HAT CON sign',y_hat)
    y_hat=np.ravel(np.sign(y_hat)).astype(int)
    print('EL Y_HAT CON RAVEL',y_hat)
    print(np.unique(y_hat,return_counts=True))
    y_train=(np.reshape(y_train,(y_train.shape[0],))).astype(int)
    #y_hat=np.array(jnp.where(y_hat < 0, -1, jnp.where(y_hat == 0, 0, 1)))
    print('el yhat100',np.count_nonzero(y_hat==y_train))
    print('el yhat100',y_hat,y_train)
    np.savetxt('y_hat.txt',y_hat)
    np.savetxt('y.txt',y_train)
    return y_hat
def get_prediction_mlp(x_train,y_train,x_v,y_v):
    
    x_train,y_train=jnp.transpose(x_train),jnp.transpose(y_train)
    layers=[x_train.shape[0],10,2]
    lr=0.01
    labels=jnp.array([0,1])
    k_clases=2
    samples=x_train.shape[0]
    y_hot=jnp.transpose(hot.one_hot(y_train,2))
    
    stop=0.0001
    max_steps=1000
    x_v,y_v,y_hot_v,samples_val=jnp.transpose(x_v),jnp.transpose(y_v),jnp.transpose(hot.one_hot(y_v,2)),x_v.shape[0]
    run_name='multilayer'
    mlp=MultilayerPerceptron(layers,lr,labels,k_clases,samples,x_train,y_train,y_hot,stop,max_steps)
    y_hat,y_hat_val=mlp.modelo_boosting(mlp.weights,max_steps,x_train,y_hot,lr,stop,x_v,y_v)
    
    return y_hat,stop,layers,lr,y_hat_val

def init_weights(x):
    w=np.ones(x.shape[0])/x.shape[0]
    return w


def error(w,y,y_hat):
    #err=np.sum(w*(np.not_equal(y,y_hat)).astype(int))
    #print('Los que son distintos',np.not_equal(y,y_hat,retunr_counts=True).astype(int))
    #err=np.sum(w * np.not_equal(y,y_hat).astype(int))/np.sum(w)
    #print('ESTE es el error',err)
    return np.sum(w * (y != y_hat))/y.shape[0]

def get_alfa(err):
    alfa=0.5*np.log((1-err)/err)
    return alfa

def update_w_correct(w,alfa,y,y_hat):
    w=w*np.exp(-alfa)
    return w
def update_w_incorrect(w,alfa,y,y_hat):
    w=w*np.exp(alfa)#(np.equal(y,y_hat)).astype(int))
    return w

def update_w(w_correct,w_incorrect,prediction,y):
    #prediction=jnp.ravel(prediction)
    prediccion_correcta=(np.equal(prediction,y)).astype(int)
    prediccion_incorrecta=(np.not_equal(prediction,y)).astype(int)

    w_correct=w_correct * prediccion_correcta
    w_incorrect=w_incorrect * prediccion_incorrecta
    pesos_finales=w_correct + w_incorrect
    #normalizamos los pesos
    pesos_finales=pesos_finales / np.sum(pesos_finales)
    return pesos_finales
def write_csv_lineal(amount_of_say,data,clasificadores):
    df_lineal=pd.DataFrame(columns=['Weak Classifier','Amount of say','Prediction'])

    for i in range(clasificadores):
        df_lineal.loc[i,'Weak Classifier']=i
        df_lineal.loc[i,'Amount of say']=amount_of_say[i]
        df_lineal.loc[i,'Prediction']=list(data[i])
    df_lineal.to_csv('AdaBoost_lineal.csv',index=False,mode='a')
    

def write_csv_stump(stump_data, amount_of_say):
    #if current depth is 0 borramos el archivo csv node_data.csv (de existir) y creamos uno nuevo con los encabezados
    #donde iremos guardando los datos de cada nodo
    df_node_data=pd.DataFrame(columns=['Stump','Amount of say','Caracteristica','Valor','Prediction div 1 igual que valor','Prediction div 2 diferente que valor'])
    df_node_data.to_csv('AdaBoost_stumps_data.csv', index=False)

    #guardamos los datos del nodo en el csv
    for i in range(len(stump_data)):
        #añadimos los datos de cada nodo al dataframe
        df_node_data=pd.DataFrame(data=stump_data[i], columns=['Nodo padre','Nodo hijo','Caracteristica','Valor','Gini impurity','div 1 No in class 1','div 1 No in class 2','div 2 No in class 1','div 2 No in class 2'], index=range(len(stump_data[i])))    

        #quitamos las colyumnas de nodo padre y nodo hijo y agregamos una columna para identificar el stump y su amount of say
        df_node_data=df_node_data.drop(columns=['Nodo padre','Nodo hijo'])
        #agregandolos al inicio del dataframe
        df_node_data.insert(0, 'Stump', i)
        df_node_data.insert(1, 'Amount of say', amount_of_say[i])

        #para div 1 y div 2 cual tiene mas elementos en cada clase y esa será la prediccion para ese stump
        div_1_no_in_class_1=df_node_data['div 1 No in class 1'].sum()
        div_1_no_in_class_2=df_node_data['div 1 No in class 2'].sum()
        div_2_no_in_class_1=df_node_data['div 2 No in class 1'].sum()
        div_2_no_in_class_2=df_node_data['div 2 No in class 2'].sum()
        if div_1_no_in_class_1 > div_1_no_in_class_2:
            df_node_data['Prediction div 1 igual que valor']=0
        else:
            df_node_data['Prediction div 1 igual que valor']=1

        if div_2_no_in_class_1 > div_2_no_in_class_2:
            df_node_data['Prediction div 2 diferente que valor']=0
        else:
            df_node_data['Prediction div 2 diferente que valor']=1

        #borramos columnas que ya no importan tanto en adaboost como gini impurity y el numero de elementos por clase
        df_node_data=df_node_data.drop(columns=['Gini impurity','div 1 No in class 1','div 1 No in class 2','div 2 No in class 1','div 2 No in class 2'])
        #guardar los datos de cada nodo en un archivo csv
        df_node_data.to_csv('AdaBoost_stumps_data.csv', mode='a', header=False, index=False)
def vector_rango(w):
    vector_rangos_pesos=[]
    for i in range(len(w)):
        if i==0:
            vector_rangos_pesos.append(w[i])
        else:
            vector_rangos_pesos.append(w[i]+ vector_rangos_pesos[-1])
    return vector_rangos_pesos
'''
def weighted_list(X,y,w):
    nuevo_X=[]
    nuevo_y=[]
    #print('el shape de w',w.shape)
    vector_rangos_pesos=vector_rango(w)
    #print('EL shapee rangos',vector_rangos_pesos.shape)
    #vector_rangos_pesos=vector_rangos_pesos.tolist()
    #print('el ultimo',vector_rangos_pesos)
    for i in range(len(X)):
        # Generar número aleatorio entre 0 y 1
        random_number = np.random.rand()

        # Encontrar el índice del primer valor en la lista que es mayor que el número aleatorio
        index_of_next_value = np.searchsorted(vector_rangos_pesos, random_number, side='right')+1
        #print('next',index_of_next_value)
        #maneja el caso especial donde el número aleatorio es mayor que el último valor en la lista
        if index_of_next_value > len(vector_rangos_pesos):
            index_of_next_value = len(vector_rangos_pesos)
        # Restar uno para obtener el índice del valor anterior
        index_of_previous_value = index_of_next_value - 1
        #print(index_of_next_value)
        # Manejar el caso especial donde el número aleatorio es menor que el primer valor en la lista
        if index_of_previous_value < 0:
            index_of_previous_value = 0

        #anexar a la nueva lista el valor de X en el índice del valor anterior
        nuevo_X.append(X[index_of_previous_value])
        nuevo_y.append(y[index_of_previous_value])

    nuevo_X=np.array(nuevo_X)
    nuevo_y=np.array(nuevo_y)
    return nuevo_X,nuevo_y
'''
def weighted_list(X,X_val,y,w,w_v):
    new_X=np.zeros((X.shape[0],X.shape[1]))
    new_X_val=np.zeros((X_val.shape[0],X_val.shape[1]))
    for i in range(X.shape[0]):
        new_X[i,:]=w[i]*X[i,:]
    for i in range(X_val.shape[0]):
        new_X_val[i,:]=w_v[i]*X_val[i,:]
    #print('El shape de x',new_X.shape)
    new_X=np.array(new_X)
    new_y=y
    return new_X,new_X_val
# creamos stumps de los datos (desicion tree con un solo nodo)
def call_model(X, y,x_v,y_v,modelo):
    w=init_weights(X)
    w_v=init_weights(x_v)
    #llamamos a nuestro desicion tree
    
    if modelo=='decision_tree':
        max_depth=1
        gini_min=0.1
        data=dt.train_tree(X,y,gini_min, max_depth)
        #hacemos la prediccion para obtener el error
        #prediction=dt.prediccion(X_to_predict=X)
        y_hat=dt.prediccion(X_to_predict=X)
        err=error(w,y,y_hat)
        #calculamos el amount of say
        alfa=get_alfa(err)
        #actualizamos los pesos
        w_correct=update_w_correct(w,alfa,y_hat,y)
        w_incorrect=update_w_incorrect(w,alfa,y_hat,y)
        w=update_w(w_correct,w_incorrect,y_hat,y)
        #new_X,new_y=weighted_list(X,y,vector_rangos_pesos)
        new_X,new_y=weighted_list(X,y,w)
        
    if modelo=='lineal':
        data=get_prediction(X,y,x_v,y_v)
        print('el shape de data',data.shape)
        err=error(w,y,data)
        #calculamos el amount of say
        alfa=get_alfa(err)
        #actualizamos los pesos
        w_correct=update_w_correct(w,alfa,y,data)
        w_incorrect=update_w_incorrect(w,alfa,y,data)
        w=update_w(w_correct,w_incorrect,data,y)
        #new_X,new_y=weighted_list(X,y,vector_rangos_pesos)
        new_X,new_y=weighted_list(X,y,w)
    
    if modelo=='mlp':
        data=get_prediction_mlp(X,y,x_v,y_v)[0]
        y_hat_val=get_prediction_mlp(X,y,x_v,y_v)[-1]
        err=error(w,y,data)
        err_v=error(w_v,y_v,y_hat_val)
        #calculamos el amount of say
        alfa=get_alfa(err)
        alfav=get_alfa(err_v)
        #actualizamos los pesos
        w_correct=update_w_correct(w,alfa,y,data)
        w_incorrect=update_w_incorrect(w,alfa,y,data)
        w=update_w(w_correct,w_incorrect,data,y)

        w_correctv=update_w_correct(w_v,alfav,y_v,y_hat_val)
        w_incorrectv=update_w_incorrect(w_v,alfav,y,y_hat_val)
        w_v=update_w(w_correctv,w_incorrectv,y_hat_val,y_v)
        #new_X,new_y=weighted_list(X,y,vector_rangos_pesos)
        #new_X,new_y=weighted_list(X,y,w)
        new_X,new_X_val=weighted_list(X,x_v,y,w,w_v)
        new_y=y

    
    return new_X,new_X_val,alfa,alfav,data,y_hat_val

def train_boosting(X, y,x_v,y_v,modelo, num_stumps):
    # creamos un vector de stumps
    sets_X=[]
    sets_y=[]
    sets_amount=[]
    model_data=[]
    for i in range(num_stumps):
        new_X,new_y,amount,data =call_model(X, y,x_v,y_v,modelo)
        sets_X.append(new_X)
        sets_y.append(new_y)
        sets_amount.append(amount)
        model_data.append(amount*data)
        X=new_X
        y=new_y
    #guardamos los datos de los stumps en un csv
    if modelo=='decision_tree':
        write_csv_stump(model_data,sets_amount)
    if modelo=='lineal':
        print('EL MODEL DATA',model_data[0])
        write_csv_lineal(sets_amount,model_data,num_stumps)
           

    return sets_X,sets_y,sets_amount

def train_boosting_lineal(X, y,x_v,y_v,modelo, num_stumps,run_name,num_iter):
    features=X.shape[1]
    experiment_name = "Ada_lineal"
    experiment = mlflow.get_experiment_by_name(experiment_name)
    experiment_id = experiment.experiment_id
    with mlflow.start_run(experiment_id=experiment_id,run_name=run_name):
        # creamos un vector de experimentos
        
        mlflow.log_param('num weak classifiers',num_stumps)
        mlflow.log_param('Iterations in gradient descent',num_iter)
        mlflow.log_param('Features',features)
        sets_X=[]
        sets_y=[]
        sets_amount=[]
        model_data=[]
        for i in range(num_stumps):
            print('Vamos en el experimento',i)
            new_X,new_y,amount,data =call_model(X, y,x_v,y_v,modelo)
            sets_X.append(new_X)
            sets_y.append(new_y)
            sets_amount.append(amount)
            model_data.append(amount*data)
            X=new_X
            y=new_y
        print(model_data)
        final=0
        for i in range(len(model_data)):
            print(model_data[i])
            final=final+sets_amount[i]*model_data[i]
        #final=np.array(jnp.where(final < 0, -1, jnp.where(final == 0, 0, 1)))
        final=np.sign(final)
        print('esta es la prediccion final',final)
        #mlflow.log_param('amount of say',sets_amount)
        accuracy=metrics.accuracy(y,final)
        recall=metrics.recall(y,final)
        precision=metrics.precision(y,final)
        print('EL ACCURACY ES',accuracy)
        mlflow.log_metric('accuracy',accuracy)
        mlflow.log_metric('precision',precision)
        mlflow.log_metric('recall',accuracy)
        
    return sets_X,sets_y,sets_amount,model_data
def train_boosting_mlp(X, y,x_v,y_v,modelo, num_stumps,run_name,num_iter):
    features=X.shape[1]
    experiment_name = "Ada_mlp"
    experiment = mlflow.get_experiment_by_name(experiment_name)
    experiment_id = experiment.experiment_id
    with mlflow.start_run(experiment_id=experiment_id,run_name=run_name):
        # creamos un vector de experimentos
        y_hat,stop,layers,lr=get_prediction_mlp(X,y,x_v,y_v)
        mlflow.log_param('num weak classifiers',num_stumps)
        mlflow.log_param('Arquitectura',layers)
        mlflow.log_param('Stop criteria',stop)
        mlflow.log_param('Features',features)
        mlflow.log_param('Learning Rate',lr)
        sets_X=[]
        sets_y=[]
        sets_amount=[]
        model_data=[]
        for i in range(num_stumps):
            print('Vamos en el experimento',i)
            new_X,new_y,amount,data =call_model(X, y,x_v,y_v,modelo)
            sets_X.append(new_X)
            sets_y.append(new_y)
            sets_amount.append(amount)
            model_data.append(data)
            X=new_X
            y=new_y
        #print(model_data)
        final=0
        for i in range(len(model_data)):
            print('este es el model data i',model_data[i])
            print('este es alfa',sets_amount[i])
            final=final+sets_amount[i]*model_data[i]
        #final=np.array(jnp.where(final < 0, -1, jnp.where(final == 0, 0, 1)))
        final=np.sign(final)
        for i in range(final.shape[0]):
            if final[i]==-1:
                final[i]=0
        print('esta es la prediccion final',final)
        np.savetxt('final.txt',final)
        #mlflow.log_param('amount of say',sets_amount)
        accuracy=metrics.accuracy(y,final)
        recall=metrics.recall(y,final)
        precision=metrics.precision(y,final)
        print('EL ACCURACY ES',accuracy)
        mlflow.log_metric('accuracy',accuracy)
        mlflow.log_metric('precision',precision)
        mlflow.log_metric('recall',recall)
        mlflow.log_metric('recall',accuracy)
        
    return sets_X,sets_y,sets_amount,model_data

def train_boosting_mlp2(X, y,x_v,y_v,modelo, num_stumps,run_name,num_iter):
    features=X.shape[1]
    experiment_name = "Ada_mlp"
    experiment = mlflow.get_experiment_by_name(experiment_name)
    experiment_id = experiment.experiment_id
    with mlflow.start_run(experiment_id=experiment_id,run_name=run_name):
        # creamos un vector de experimentos
        y_hat,stop,layers,lr,y_hat_val=get_prediction_mlp(X,y,x_v,y_v)
        mlflow.log_param('num weak classifiers',num_stumps)
        mlflow.log_param('Arquitectura',layers)
        mlflow.log_param('Stop criteria',stop)
        mlflow.log_param('Features',features)
        mlflow.log_param('Learning Rate',lr)
        sets_X=[]
        sets_y=[]
        sets_amount=[]
        model_data=[]
        sets_Xv=[]
        sets_yv=[]
        sets_amountv=[]
        model_datav=[]
        for i in range(num_stumps):
            print('Vamos en el experimento',i)
            new_X,new_xv,amount,amountv,data,y_hat_val =call_model(X, y,x_v,y_v,modelo)
            sets_X.append(new_X)
            sets_Xv.append(new_xv)
            sets_amount.append(amount)
            model_data.append(data)
            sets_amountv.append(amountv)
            model_datav.append(y_hat_val)
            X=new_X
            x_v=new_xv
        #print(model_data)
        final=0
        finalv=0
        for i in range(len(model_data)):
            print('este es el model data i',model_data[i])
            print('este es alfa',sets_amount[i])
            final=final+sets_amount[i]*model_data[i]
            finalv=finalv+sets_amountv[i]*model_datav[i]
        #final=np.array(jnp.where(final < 0, -1, jnp.where(final == 0, 0, 1)))
        final=np.sign(final)
        finalv=np.sign(finalv)
        for i in range(final.shape[0]):
            if final[i]==-1:
                final[i]=0
        for i in range(finalv.shape[0]):
            if finalv[i]==-1:
                finalv[i]=0
        print('esta es la prediccion final',final)
        np.savetxt('final.txt',final)
        #mlflow.log_param('amount of say',sets_amount)
        accuracy=metrics.accuracy(y,final)
        recall=metrics.recall(y,final)
        precision=metrics.precision(y,final)
        print('EL ACCURACY ES',accuracy)
        mlflow.log_metric('accuracy',accuracy)
        mlflow.log_metric('precision',precision)
        mlflow.log_metric('recall',recall)
        mlflow.log_metric('recall',accuracy)

        accuracyv=metrics.accuracy(y_v,finalv)
        recallv=metrics.recall(y_v,finalv)
        precisionv=metrics.precision(y_v,finalv)
        print('EL ACCURACY de val ES',accuracyv)
        mlflow.log_metric('accuracy validation',accuracyv)
        mlflow.log_metric('precision validation',precisionv)
        mlflow.log_metric('recall validation',recallv)
        mlflow.log_metric('recall validation',accuracyv)
        
    return sets_X,sets_y,sets_amount,model_data

def prediction(X_to_predict,modelo):
    if modelo=='decision_tree':
        #read the csv with the adaboost trained sin indices
        df=pd.read_csv('AdaBoost_stumps_data.csv', index_col=False)
        #get the number of stumps
        num_stumps=len(df['Stump'].unique())
        prediccion_final=[]
        for i in range(len(X_to_predict)):
            row_to_predict=X_to_predict[i]
            prediccion_stumps=[]
            amount_of_say_stumps=[]
            for stump in range(num_stumps):
                df_stump=df[df['Stump']==stump]
                caracteristica=df_stump['Caracteristica'].values[0]
                valor=df_stump['Valor'].values[0]
                if np.isin(np.array(row_to_predict[caracteristica]),valor):
                    prediccion_stumps.append(df_stump['Prediction div 1 igual que valor'].values[0])
                    amount_of_say_stumps.append(df_stump['Amount of say'].values[0])
                else:
                    prediccion_stumps.append(df_stump['Prediction div 2 diferente que valor'].values[0])
                    amount_of_say_stumps.append(df_stump['Amount of say'].values[0])
        
            #sumamos los amount of say de acuerdo a la prediccion
            amount_of_say_clase_1=0
            amount_of_say_clase_2=0
            for i in range(len(prediccion_stumps)):
                if prediccion_stumps[i]==0:
                    amount_of_say_clase_1+=amount_of_say_stumps[i]
                else:
                    amount_of_say_clase_2+=amount_of_say_stumps[i]
            #comparamos los amount of say y hacemos la prediccion
            if amount_of_say_clase_1 > amount_of_say_clase_2:
                prediccion_final.append(0)
            else:
                prediccion_final.append(1)
    if modelo=='lineal':
        df=pd.read_csv('AdaBoost_lineal.csv',index_col=False)
        y=df['Prediction']
        print(y)
        #df['Prediction'] = df['Prediction'].apply(lambda x: np.array2string(x, separator=','))
        df['Prediction_np'] = df['Prediction'].apply(lambda x: np.array(x))
        df['Prediction'] = df['Prediction'].apply(lambda x: ast.literal_eval(x))
        print(y)
        alfa=df['Amount of say']
        print(alfa)
        df['Prediction'] = df['Prediction'].apply(lambda x: np.multiply(x, alfa[df.index[x.name]]))
        #prediccion_final = np.sign(np.sum([np.array(row) for row in y.values], axis=0))
        


    return prediccion_final

if __name__=="__MAIN__":
    train_boosting()
    prediction()










