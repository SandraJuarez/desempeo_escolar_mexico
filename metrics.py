import jax.numpy as jnp
import jax
from jax import jit
#------Metricas---------------- 

def check_jaxarray(y,y_hat):
    if isinstance(y,jnp.ndarray)==False:
        y=jnp.asarray(y)
        y_hat=jnp.asarray(y_hat)
    

def precision(y, y_hat): 
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
def recall(y, y_hat): 
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

def accuracy(y, y_hat): 
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

if __name__=="__MAIN__":
    precision()
    accuracy()
    recall()