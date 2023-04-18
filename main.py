import clean_database
import DataAugmentation
import Feature_Engineering as fe
import train_test_validation as ttv
#import graficar
#from mlp import MultilayerPerceptron
import numpy as np
import jax.numpy as jnp
import y_hot as hot
from mlp_mlflow import MultilayerPerceptron
import mlflow


datos_bajos,datos=clean_database.Clean_Data()
datos_aumentados=DataAugmentation.Smote(datos_bajos)
print(datos_aumentados.shape)


datos=np.concatenate((datos,datos_aumentados),axis=0)
datos=datos[np.random.permutation(datos.shape[0]),:]
x,y=datos[:,1:],datos[:,0]
#x=x[:,np.random.permutation(x.shape[1])]
print('este es y',y)
np.savetxt('datosy.csv',datos,delimiter=',')
x1,x2=fe.separate(x,y) #tenemos que separar los features en
print('estas son las muestras en casda clase',x1.shape[0],x2.shape[0])
size=x1.shape[1]
min_group=90 #son los features con los que nos queremos quedar
x1,x2,ganadores=fe.sequential_backwards(size,x1,x2,min_group)
x=x[:,ganadores]
x_train, x_test, x_v,y_train, y_test, y_v=ttv.split_data(x,y)

##llamamos al model de multilayer perceptron
x_train,y_train=jnp.transpose(x_train),jnp.transpose(y_train)
layers=[89,30,30,2]
lr=0.01
labels=jnp.array([0,1])
k_clases=2
samples=x.shape[0]
y_hot=jnp.transpose(hot.one_hot(y_train,2))
stop=0.0001
max_steps=10000
x_val,y_val,y_hot_val,samples_val=jnp.transpose(x_v),jnp.transpose(y_v),jnp.transpose(hot.one_hot(y_v,2)),x_v.shape[0]
run_name='mlp'
mlp_mlflow=MultilayerPerceptron(layers,lr,labels,k_clases,samples,x_train,y_train,y_hot,stop,max_steps)
loss,recall_list,precision_list,loss_list,precision_list_val,recall_list_val,loss_list_val,run.info.experiment_id, run.info.run_id=mlp_mlflow.modelo(mlp_mlflow.weights,max_steps,x_train,y_train,y_hot,lr,k_clases,samples,labels,stop,x_val,y_val,y_hot_val,samples_val,run_name)
