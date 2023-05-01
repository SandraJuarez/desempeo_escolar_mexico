import pandas as pd
def Clean_Data():

    dfP=pd.read_csv('datos.csv')
    dfv=pd.read_csv('vivienda.csv')

    # merge dataframes based on id column
    dfd = pd.merge(dfv, dfP, on='FOLIO', how='inner')

    #dfd = dfd.dropna(thresh=dfd.shape[1]-15)
    values_nivel=[1,2,3,10,11,12] 
    values_nivel2=[1,2,9,10,11]
    dfd=dfd[dfd.PA3_3_NIVEL.isin(values_nivel)==False] #quitamos los rows de los que van en kínder y primaria
    dfd=dfd[dfd.PA3_3_MODMAT!=1] #quitamos los rows de los que estuvieron en maternal
    dfd=dfd[dfd.PB3_5_MODMAT!=1] #quitamos los rows de los que se inscribieron a maternal
    dfd=dfd[dfd.PB3_5_NIVEL.isin(values_nivel2)==False]#quitamos a los que se inscribieron al kinder y a los que se inscribieron a maestría/doctorado
    dfd=dfd[dfd.PC3_1!=2]
    values_c=[1,2,5,6,7,8,9,10,11,12,13,14] #quitamos los que por motivos personales no se inscribieron
    dfd=dfd[dfd.PC3_6.isin(values_c)==False]
    #dfd=dfd[dfd.FILTRO_C!=1] #Quitamos a los que no estuvieron inscritos en el anterior ni en el siguiente
    dfd=dfd[dfd.PA3_6!=4]
    dfd[dfd==9]=0
    dfd[dfd==99]=0
    dfd[dfd==98]=0

    dfd.drop(columns=['PA3_3_MODMAT','PB3_5_MODMAT','FOLIO'],inplace=True)
    dfd['FILTRO_A']=dfd['FILTRO_A'].fillna(0) #SE CAMBIÓ DE ESCUELA
    dfd['FILTRO_B']=dfd['FILTRO_B'].fillna(0) #MENORES DE 17?
    dfd['FILTRO_C']=dfd['FILTRO_C'].fillna(0) #ABANDONO ESCOLAR?
    dfd['PA3_5']=dfd['PA3_5'].fillna(0) #razones por las que no concluyó
    dfd['PB3_4']=dfd['PB3_4'].fillna(0) #RAZPON POR LA QUE SE CAMBIÓ DE ESCUELA
    dfd[['PA3_3_NIVEL','PB3_5_NIVEL','PA3_3_BIMESTRE','PA3_3_TRIMESTRE','PA3_3_CUATRIMESTRE','PA3_3_SEMESTRE','PA3_3_ANIO','PB3_5_BIMESTRE','PB3_5_TRIMESTRE','PB3_5_CUATRIMESTRE','PB3_5_SEMESTRE','PB3_5_ANIO']]=dfd[['PA3_3_NIVEL','PB3_5_NIVEL','PA3_3_BIMESTRE','PA3_3_TRIMESTRE','PA3_3_CUATRIMESTRE','PA3_3_SEMESTRE','PA3_3_ANIO','PB3_5_BIMESTRE','PB3_5_TRIMESTRE','PB3_5_CUATRIMESTRE','PB3_5_SEMESTRE','PB3_5_ANIO']].fillna(0)
    dfd=dfd.fillna('NaN')

    dfd.drop(columns=['P3_2'],inplace=True)

    #estas son las que usaremos para medir mal desempeño
    a=dfd['PA3_6'].to_numpy()
    b=dfd['PA3_7_2'].to_numpy()
    c=dfd['PA3_7_3'].to_numpy()
    d=dfd['PA3_4'].to_numpy() #SON LOS QUE NO TERMINARON EL AÑO POR BAJO DESEMPEÑO
    e=dfd['PB3_2'].to_numpy() #DEJAMOS LOS QUE RESPONDIERON QUE YA NO SE INSCRIBIERON AL SIGUIENTE POR BAJO DESEMPEÑO
    f=dfd['PC3_6'].to_numpy() #los que dejaron la escuela porque tuvieron bajo desempeño


    dfd.drop(columns=['PB3_7','PB3_8'],inplace=True) #QUITAMOS ESTAS DOS PORQUE LES FALTA LA MITAD DE LA DATA
    #EN ESTAS DOS de arriba PREGUNTABAN CUÁNTOS DÍAS VAN A LA ESCUEL Y POR QUÉ
    dfd.drop(columns=['NIVEL_A','GRADO_A','NIVEL_B','GRADO_B','ENT_y'],inplace=True)#estas preguntas estaban repetidas
    dfd.drop(columns=['FACTOR_x','FACTOR_y','N_REN'],inplace=True) #ESTAS COLUMNAS NO SE SABE QUÉ SON, NO LO INDICARON EN EL CATÁLOGO
    
    import numpy as np
    total=d.shape[0]
    clase=np.zeros(total)
    contador =0
    contadornans=0
    for i in range(total):
        if a[i]==1 or a[i]==2:
            clase[i]=1
            
        elif b[i]==1:
            clase[i]=1
            
        elif c[i]==1:
            clase[i]=1
            
        elif d[i]==2:
            clase[i]=1
            
        elif e[i]==3 or e[i]==5:
            
            clase[i]=1
        elif f[i]==3 or f[i]==4:
            clase[i]=1
        elif a[i]==b[i]==c[i]==d[i]==e[i]==f[i]=='NaN':
            clase[i]='NaN'
            contadornans+=1
        else:
        
            clase[i]=0
        

    
    dfd.insert(0,'clase',clase)

    dfd['PB3_4'] = dfd['PB3_4'].fillna(0) #los que se cambiaron de escuela
    #porque a los mayores de 17 ya no les preguntaron las siguientes (las de si los ayudan los papás)
    dfd.loc[dfd['EDAD'] >= 17, 'PB3_13_7'] = 0
    dfd.loc[dfd['EDAD'] >= 17, 'PB3_13_6'] = 0
    dfd.loc[dfd['EDAD'] >= 17, 'PB3_13_5'] = 0
    dfd.loc[dfd['EDAD'] >= 17, 'PB3_13_4'] = 0
    dfd.loc[dfd['EDAD'] >= 17, 'PB3_13_3'] = 0
    dfd.loc[dfd['EDAD'] >= 17, 'PB3_13_2'] = 0
    dfd.loc[dfd['EDAD'] >= 17, 'PB3_13_1'] = 0
    dfd.loc[dfd['EDAD'] >= 17, 'PB3_14'] = 0
    dfd.loc[dfd['EDAD'] >= 17, 'PB3_15'] = 0

    #LOS QUE NO CUMPLEN CON EL FILTRO C, O SEA QUE SI ESTUVIERON INSCRITOS EN EL ANTERIOR Y SIGUEINTE
    dfd.loc[dfd['FILTRO_C']==0,['PC3_1','PC3_3_1','PC3_3_2','PC3_4','PC3_5','PC3_6','PC3_7','PC3_8']]=0
    dfd.loc[dfd['FILTRO_C']=='NaN',['PC3_1','PC3_3_1','PC3_3_2','PC3_4','PC3_5','PC3_6','PC3_7','PC3_8']]=0
    #PARA LOS QUE RETOMARÁN LA ESCUELA
    dfd.loc[dfd['FILTRO_C']==2,['PC3_1','PC3_3_1','PC3_3_2','PC3_4','PC3_5','PC3_6','PC3_7']]=0

    #PARA LOS QUE RESPONDIERON QUE NO TRABAJAN, LAS OTRAS PREGUNTAS NO APLICAN
    dfd.loc[dfd['FILTRO_D']==2,['PD3_1','PD3_2','PD3_3']]=0
    dfd.loc[(dfd['FILTRO_D']=='NaN') & (dfd['EDAD']<=14),['PD3_1','PD3_2','PD3_3']]=0

    grupos=dfd.groupby(['clase'])


    d1=pd.DataFrame(grupos.get_group(1))

    d2=pd.DataFrame(grupos.get_group(0))
    

    #rellenamos los valores faltantes donde preguntan si le ayudaron.
    #si respondieron que nadie les ayuda, rellenamos con un 2 las otras preguntas
    d1.loc[d1['PB3_13_7'] == 1, 'PB3_13_6'] = 2
    d1.loc[d1['PB3_13_7'] == 1, 'PB3_13_5'] = 2
    d1.loc[d1['PB3_13_7'] == 1, 'PB3_13_4'] = 2
    d1.loc[d1['PB3_13_7'] == 1, 'PB3_13_3'] = 2
    d1.loc[d1['PB3_13_7'] == 1, 'PB3_13_2'] = 2
    d1.loc[d1['PB3_13_7'] == 1, 'PB3_13_1'] = 2

    #para los que no concluyeron en algunos casos ya no preguntaron sobre los featues que seguían

    ###################################################################################
    #####################################################################################
    ######################################################################################
    #POR SI NO PREGUNTARON A LOS QUE NO CONCLUYERON LAS PARTES DE COMO SE INGRESA A LA ESCUELA
    d1.loc[d1['PA3_4']==2,'PB3_6']=0 #asistía a presencial ono?
    d1.loc[d1['PA3_4']==2,'PB3_3']=0 #la escuela es privada o no?
    d1.loc[d1['PA3_4']==2,'PA3_7_1']=0 #RECIBIÓ ASESORÍAS EXTRA?
    d1.loc[d1['PA3_4']==2,'PA3_2']=0 #PRIVADA?
    #METODOS DE EVALUACIÓN
    d1.loc[(d1['PA3_4']==2),
        #(d1['PA3_8_1']=='NaN')&
        #(d1['PA3_8_2']=='NaN')&
        #(d1['PA3_8_3']=='NaN')&
        #(d1['PA3_8_5']=='Nan')&
        #(d1['PA3_8_6']=='NaN')&
        #(d1['PA3_8_7']=='NaN')&
        #(d1['PA3_8_8']=='NaN'),
        ['PA3_8_1','PA3_8_2','PA3_8_3','PA3_8_4','PA3_8_5','PA3_8_6','PA3_8_7','PA3_8_8']]=0


    #PREGUNTAS SOBRE METODOS DE INGRESO A LA ESCUELA
    d1.loc[(d1['PA3_4']==2)&
        (d1['PB3_9_1']=='NaN')&
        (d1['PB3_9_2']=='NaN')&
        (d1['PB3_9_3']=='NaN'),
        ['PB3_9_1','PB3_9_2','PB3_9_3']]=0

    #MATERIAL QUE USAN SUS MAESTROS
    d1.loc[(d1['PA3_4']==2)&
        (d1['PB3_10_1']=='NaN')&
        (d1['PB3_10_2']=='NaN')&
        (d1['PB3_10_3']=='NaN')&
        (d1['PB3_10_4']=='NaN')&
        (d1['PB3_10_5']=='NaN'),
        ['PB3_10_1','PB3_10_2','PB3_10_3','PB3_10_4','PB3_10_5']]=0

    #MEDIOS PARA INFORMARLE DE SUS TAREAS
    d1.loc[(d1['PA3_4']==2)&
        (d1['PB3_11_1']=='NaN')&
        (d1['PB3_11_2']=='NaN')&
        (d1['PB3_11_3']=='NaN')&
        (d1['PB3_11_4']=='NaN')&
        (d1['PB3_11_5']=='NaN'),
        ['PB3_11_1','PB3_11_2','PB3_11_3','PB3_11_4','PB3_11_5']]=0

    #MATERIALES QUE USA PARA SUS TAREAS
    d1.loc[(d1['PA3_4']==2)&
        (d1['PB3_12_1']=='NaN')&
        (d1['PB3_12_2']=='NaN')&
        (d1['PB3_12_3']=='NaN')&
        (d1['PB3_12_4']=='NaN')&
        (d1['PB3_12_5']=='NaN')&
        (d1['PB3_12_6']=='NaN')&
        (d1['PB3_12_7']=='NaN')&
        (d1['PB3_12_8']=='NaN'),
        ['PB3_12_1','PB3_12_2','PB3_12_3','PB3_12_4','PB3_12_5','PB3_12_6','PB3_12_7','PB3_12_8']]=0

    #¿LA PERSONA RECIBIO AYUDA de la familia?
    d1.loc[(d1['PA3_4']==2) &
        (d1['PB3_13_1']=='NaN')&
        (d1['PB3_13_2']=='NaN')&
        (d1['PB3_13_3']=='NaN')&
        (d1['PB3_13_4']=='NaN')&
        (d1['PB3_13_5']=='NaN')&
        (d1['PB3_13_6']=='NaN')&
        (d1['PB3_13_7']=='NaN')&
        (d1['PB3_15']=='NaN')&
        (d1['PB3_14']=='NaN'),
        ['PB3_13_1','PB3_13_2','PB3_13_3','PB3_13_4','PB3_13_5','PB3_13_6','PB3_13_7','PB3_14','PB3_15']]=0

    #¿cómo se siente de salud la persona?

    d1.loc[(d1['PA3_4']==2)&
        (d1['PB3_16_1']=='NaN')&
        (d1['PB3_16_2']=='NaN')&
        (d1['PB3_16_3']=='NaN')&
        (d1['PB3_16_4']=='NaN')&
        (d1['PB3_16_5']=='NaN'),
        ['PB3_16_1','PB3_16_2','PB3_16_3','PB3_16_4','PB3_16_5']]=0

    #####################################################################################################
    #################################################################################################
    ###################################################################################################
    d1.loc[d1['FILTRO_C']==1,'PA3_2']=0 #la escuela es privada o no
    d1.loc[d1['FILTRO_C']==1,'PB3_3']=0 #la escuela es privada o no?
    d1.loc[d1['FILTRO_C']==1,'PB3_6']=0 #asiste a presencial ono?
    d1.loc[d1['FILTRO_C']==1,'PC3_8']=0 
    d1.loc[d1['FILTRO_C']==1,'PA3_7_1']=0 #RECIBIÓ ASESORÍAS EXTRA?
    #METODOS DE EVALUACIÓN
    d1.loc[(d1['FILTRO_C']==1),
        #(d1['PA3_8_1']=='NaN')&
        #(d1['PA3_8_2']=='NaN')&
        #(d1['PA3_8_3']=='NaN')&
        #(d1['PA3_8_5']=='Nan')&
        #(d1['PA3_8_6']=='NaN')&
        #(d1['PA3_8_7']=='NaN')&
        #(d1['PA3_8_8']=='NaN'),
        ['PA3_8_1','PA3_8_2','PA3_8_3','PA3_8_4','PA3_8_5','PA3_8_6','PA3_8_7','PA3_8_8']]=0


    #PREGUNTAS SOBRE METODOS DE INGRESO A LA ESCUELA
    d1.loc[(d1['FILTRO_C']==1) &
        (d1['PB3_9_1']=='NaN')&
        (d1['PB3_9_2']=='NaN')&
        (d1['PB3_9_3']=='NaN'),
        ['PB3_9_1','PB3_9_2','PB3_9_3']]=0

    #MATERIAL QUE USAN SUS MAESTROS
    d1.loc[(d1['FILTRO_C']==1) &
        (d1['PB3_10_1']=='NaN')&
        (d1['PB3_10_2']=='NaN')&
        (d1['PB3_10_3']=='NaN')&
        (d1['PB3_10_4']=='NaN')&
        (d1['PB3_10_5']=='NaN'),
        ['PB3_10_1','PB3_10_2','PB3_10_3','PB3_10_4','PB3_10_5']]=0

    #MEDIOS PARA INFORMARLE DE SUS TAREAS
    d1.loc[(d1['FILTRO_C']==1)&
        (d1['PB3_11_1']=='NaN')&
        (d1['PB3_11_2']=='NaN')&
        (d1['PB3_11_3']=='NaN')&
        (d1['PB3_11_4']=='NaN')&
        (d1['PB3_11_5']=='NaN'),
        ['PB3_11_1','PB3_11_2','PB3_11_3','PB3_11_4','PB3_11_5']]=0

    #MATERIALES QUE USA PARA SUS TAREAS
    d1.loc[(d1['FILTRO_C']==1)&
        (d1['PB3_12_1']=='NaN')&
        (d1['PB3_12_2']=='NaN')&
        (d1['PB3_12_3']=='NaN')&
        (d1['PB3_12_4']=='NaN')&
        (d1['PB3_12_5']=='NaN')&
        (d1['PB3_12_6']=='NaN')&
        (d1['PB3_12_7']=='NaN')&
        (d1['PB3_12_8']=='NaN'),
        ['PB3_12_1','PB3_12_2','PB3_12_3','PB3_12_4','PB3_12_5','PB3_12_6','PB3_12_7','PB3_12_8']]=0

    #¿LA PERSONA RECIBIO AYUDA de la familia?
    d1.loc[(d1['FILTRO_C']==1) &
        (d1['PB3_13_1']=='NaN')&
        (d1['PB3_13_2']=='NaN')&
        (d1['PB3_13_3']=='NaN')&
        (d1['PB3_13_4']=='NaN')&
        (d1['PB3_13_5']=='NaN')&
        (d1['PB3_13_6']=='NaN')&
        (d1['PB3_13_7']=='NaN')&
        (d1['PB3_15']=='NaN')&
        (d1['PB3_14']=='NaN'),
        ['PB3_13_1','PB3_13_2','PB3_13_3','PB3_13_4','PB3_13_5','PB3_13_6','PB3_13_7','PB3_14','PB3_15']]=0

    #¿cómo se siente de salud la persona?

    d1.loc[(d1['FILTRO_C']==1)&
        (d1['PB3_16_1']=='NaN')&
        (d1['PB3_16_2']=='NaN')&
        (d1['PB3_16_3']=='NaN')&
        (d1['PB3_16_4']=='NaN')&
        (d1['PB3_16_5']=='NaN'),
        ['PB3_16_1','PB3_16_2','PB3_16_3','PB3_16_4','PB3_16_5']]=0

    ###################################################################################
    #####################################################################################
    ######################################################################################
    #METODOS DE EVALUACIÓN
    d1.loc[d1['FILTRO_C']==2,'PA3_2']=0 #la escuela es privada o no?
    d1.loc[d1['FILTRO_C']==2,'PB3_3']=0 #la escuela es privada o no?
    d1.loc[d1['FILTRO_C']==2,'PA3_7_1']=0 #RECIBIÓ ASESORÍAS EXTRA?
    d1.loc[d1['FILTRO_C']==2,'PB3_6']=0 #asiste a presencial o no?

    d1.loc[(d1['FILTRO_C']==2)&
        (d1['PA3_8_1']=='NaN')&
        (d1['PA3_8_2']=='NaN')&
        (d1['PA3_8_3']=='NaN')&
        (d1['PA3_8_5']=='Nan')&
        (d1['PA3_8_6']=='NaN')&
        (d1['PA3_8_7']=='NaN')&
        (d1['PA3_8_8']=='NaN'),
        ['PA3_8_1','PA3_8_2','PA3_8_3','PA3_8_4','PA3_8_5','PA3_8_6','PA3_8_7','PA3_8_8']]=0


    #PREGUNTAS SOBRE METODOS DE INGRESO A LA ESCUELA
    d1.loc[(d1['FILTRO_C']==2) &
        (d1['PB3_9_1']=='NaN')&
        (d1['PB3_9_2']=='NaN')&
        (d1['PB3_9_3']=='NaN'),
        ['PB3_9_1','PB3_9_2','PB3_9_3']]=0

    #MATERIAL QUE USAN SUS MAESTROS
    d1.loc[(d1['FILTRO_C']==2) &
        (d1['PB3_10_1']=='NaN')&
        (d1['PB3_10_2']=='NaN')&
        (d1['PB3_10_3']=='NaN')&
        (d1['PB3_10_4']=='NaN')&
        (d1['PB3_10_5']=='NaN'),
        ['PB3_10_1','PB3_10_2','PB3_10_3','PB3_10_4','PB3_10_5']]=0

    #MEDIOS PARA INFORMARLE DE SUS TAREAS
    d1.loc[(d1['FILTRO_C']==2)&
        (d1['PB3_11_1']=='NaN')&
        (d1['PB3_11_2']=='NaN')&
        (d1['PB3_11_3']=='NaN')&
        (d1['PB3_11_4']=='NaN')&
        (d1['PB3_11_5']=='NaN'),
        ['PB3_11_1','PB3_11_2','PB3_11_3','PB3_11_4','PB3_11_5']]=0

    #MATERIALES QUE USA PARA SUS TAREAS
    d1.loc[(d1['FILTRO_C']==2)&
        (d1['PB3_12_1']=='NaN')&
        (d1['PB3_12_2']=='NaN')&
        (d1['PB3_12_3']=='NaN')&
        (d1['PB3_12_4']=='NaN')&
        (d1['PB3_12_5']=='NaN')&
        (d1['PB3_12_6']=='NaN')&
        (d1['PB3_12_7']=='NaN')&
        (d1['PB3_12_8']=='NaN'),
        ['PB3_12_1','PB3_12_2','PB3_12_3','PB3_12_4','PB3_12_5','PB3_12_6','PB3_12_7','PB3_12_8']]=0

    #¿LA PERSONA RECIBIO AYUDA de la familia?
    d1.loc[(d1['FILTRO_C']==2) &
        (d1['PB3_13_1']=='NaN')&
        (d1['PB3_13_2']=='NaN')&
        (d1['PB3_13_3']=='NaN')&
        (d1['PB3_13_4']=='NaN')&
        (d1['PB3_13_5']=='NaN')&
        (d1['PB3_13_6']=='NaN')&
        (d1['PB3_13_7']=='NaN')&
        (d1['PB3_15']=='NaN')&
        (d1['PB3_14']=='NaN'),
        ['PB3_13_1','PB3_13_2','PB3_13_3','PB3_13_4','PB3_13_5','PB3_13_6','PB3_13_7','PB3_14','PB3_15']]=0

    #¿cómo se siente de salud la persona?

    d1.loc[(d1['FILTRO_C']==2)&
        (d1['PB3_16_1']=='NaN')&
        (d1['PB3_16_2']=='NaN')&
        (d1['PB3_16_3']=='NaN')&
        (d1['PB3_16_4']=='NaN')&
        (d1['PB3_16_5']=='NaN'),
        ['PB3_16_1','PB3_16_2','PB3_16_3','PB3_16_4','PB3_16_5']]=0


    #d1.loc[(d1['PA3_4']==2)&(d1.loc[:,'PA3_8_1':'PA3_8_7']=='NaN'),'PA3_8_1':'PA3_8_7']=0
    d1.drop(columns=['PA3_6','PA3_7_2','PA3_7_3','PA3_4','PB3_2','PC3_6'],inplace=True)
    d2.drop(columns=['PA3_6','PA3_7_2','PA3_7_3','PA3_4','PB3_2','PC3_6'],inplace=True)
    d1=d1.replace('NaN',np.nan)
    d2=d2.replace('NaN',np.nan)
    d1= d1.dropna(thresh=d1.shape[1]-5)
    d2= d2.dropna(thresh=d2.shape[1]-5)
    d1=d1.fillna(0)
    d2=d2.iloc[:6239]
    d2=d2.fillna(0)
    print(d1.shape[0],d2.shape[0])

    dfd=d1.append(d2)
    #normalizamos
    features=int(dfd.shape[1])
    for j in dfd.columns[1:]:
        dfd[j] = dfd[j] /dfd[j].abs().max()

    for j in d1.columns[1:]:
        d1[j] = d1[j] /d1[j].abs().max()

    
    dfd.to_csv('limpios.csv')
    datos=dfd[dfd.columns[0:]].to_numpy()
    header=list(dfd.columns)
    x=dfd[dfd.columns[1:]].to_numpy()
    y=dfd[dfd.columns[0]].to_numpy
    datos_bajo = d1[d1.columns[0:]].to_numpy()
    
    return datos_bajo,datos,header

if __name__=="__MAIN__":
    Clean_Data()