import sys
sys.path.append('./models/')
from Arima import VAR_m
from SIRH  import SEIR_model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from evaluate_model import WIS
import json
import warnings
warnings.filterwarnings('ignore')

if __name__ =='__main__':
    args = sys.argv # the arguments give the pandemic on which evaluate the models
    mob_of_the_pandemic=(args[1])
    number_of_the_pandemic=(args[2])
    frequency=int(args[3])
    path_to_file='all_pandemics/pandemic_'+str(mob_of_the_pandemic)+'_'+str(number_of_the_pandemic)+'.csv'
    print(path_to_file)
    df=pd.read_csv(path_to_file)

    df.index=['n_hospitalized', 'n_infectious', 'mobility', 'R_eff']
    df.drop(columns=['Unnamed: 0'], inplace=True)

    n_hospitalized=np.array(df.loc['n_hospitalized'])
    n_infectious=np.array(df.loc['n_infectious'])
    mobility=np.array(df.loc['mobility'])

    indices = np.arange(0, len(n_infectious), frequency)
    selected_values = n_infectious[indices]
    new_indices = np.arange(len(n_infectious))
    n_infectious_n = np.interp(new_indices, indices, selected_values)

    indices = np.arange(0, len(mobility), frequency)
    selected_values = mobility[indices]
    new_indices = np.arange(len(mobility))
    mobility_n = np.interp(new_indices, indices, selected_values)

    myvar=VAR_m()
    myseir=SEIR_model()
    alphas=np.array([0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    indexs_points=[[20*i] for i in range(1, 15) ] #15
    weights=np.concatenate((np.array([0.5]), alphas * 0.5))
    reach=7

    models3Dnames=[ 'VAR','SEIR Mob']
    models3D=[myvar, myseir]

    #models3Dnames=[ 'VAR']
    #models3D=[myvar]

    dico_wis=dict()
    for point in indexs_points:
        dico_wis[str(point)]=[]
    dico_rmse=dict()
    for point in indexs_points:
        dico_rmse[str(point)]=[]

    for point in indexs_points:
        print(point)

        prediction_7=[]
        prediction_14=[]

        prediction_7_quant=[]
        prediction_14_quant=[]

        ref_pred=np.inf
        ref_quant=np.inf
        ref_wis=np.inf
        ref_rmse=np.inf

        for index, model in enumerate(models3D):
                try:
                    m=indices[int(np.floor(point[0]/frequency))]
                    n_infectious_np = n_infectious_n.copy()
                    mobility_np = mobility_n.copy()
                    n_infectious_np[m:]=n_infectious_np[m-1]
                    mobility_np[m:]=mobility_np[m-1]
                    data3D=np.array([n_hospitalized, n_infectious_np, mobility_np])

                    model.train(train_dates = [i for i in range(point[0])], data = data3D[:,:point[0]])
                    error_on_training=False
                except:
                    dico_wis[str(point)].append(ref_wis)
                    dico_rmse[str(point)].append(ref_rmse)
                    error_on_training=True
                    print("Error in training for " + models3Dnames[index])
                if not error_on_training :
                    intervals=[]
                    for alpha in alphas:
                        try:
                            prediction, interval = model.predict(reach, alpha)
                            interval_low=interval[0][-1]
                            interval_high=interval[1][-1]
                            prediction=prediction[-1]
                        except :
                            interval_low=0
                            interval_high=0
                            prediction=np.inf
                        intervals.append((interval_low, interval_high))
                    if prediction == np.inf or np.isnan(prediction):
                        wis=ref_wis
                        RMSE=ref_rmse
                    else :
                        try :
                            wis=WIS(prediction=prediction, intervals = intervals, point_of_evaluation = n_hospitalized[point[0]+reach-1], alphas = alphas , weights = weights)
                            if np.isnan(wis):
                                wis=ref_wis
                        except :
                            wis=ref_wis
                        try :
                            RMSE=np.sqrt((prediction - n_hospitalized[point[0]+reach-1])**2)
                            if np.isnan(RMSE):
                                RMSE=ref_rmse
                        except :
                            RMSE=ref_rmse
                    dico_wis[str(point)].append(wis)
                    dico_rmse[str(point)].append(RMSE)

    with open('./results/thinning_evaluation/evaluation_with_RMSE_on_pandemic_'+str(mob_of_the_pandemic)+'_'+str(number_of_the_pandemic)+'_and_reach_='+str(reach)+'_freq_='+str(frequency)+'.json', 'w') as f:
            json.dump(dico_rmse, f)
    print('test')
    with open('./results/thinning_evaluation/evaluation_with_WIS_on_pandemic_'+str(mob_of_the_pandemic)+'_'+str(number_of_the_pandemic)+'_and_reach_='+str(reach)+'_freq_='+str(frequency)+'.json', 'w') as f:
            json.dump(dico_wis, f)
