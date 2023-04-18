import matplotlib.pyplot as plt
def graficar_pr(recall_list,precision_list,recall_list_v,precision_list_v):
        plt.style.use('rose-pine')
        plt.plot(recall_list,precision_list,color='#fb9f9f',marker='*',label='Training')
        plt.plot(recall_list_v,precision_list_v,color='#99ccff.',marker='*',label='Validaton')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall curve')
        #plt.show()

def graficar_rc(loss_list,loss_list_v):
    plt.style.use('rose-pine')
    plt.plot(loss_list,color='#fb9f9f',label='Training')
    plt.plot(loss_list_v,color='#99ccff',label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Rate of Convergency')
    #plt.show()

def graficar_pr_ridge(recall_list,precision_list,recall_list_v,precision_list_v,lbda):
    plt.style.use('rose-pine')
    plt.plot(lbda,precision_list,color='#fb9f9f',label='Training')
    plt.plot(lbda,recall_list,color='#fb9f9f',label='Training')
    plt.plot(lbda,precision_list_v,color='#99ccff',label='Validation')
    plt.plot(lbda,recall_list_v,color='#99ccff',label='Validation')
    plt.xlabel('Lambda')
    plt.ylabel('Loss')
    plt.title('Loss function')

if __name__=="__MAIN__":
      graficar_pr()
      graficar_rc()
      graficar_pr_ridge()