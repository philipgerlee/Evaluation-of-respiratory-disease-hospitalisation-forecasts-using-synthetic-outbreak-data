import sys
sys.path.append('./models/')
from Arima import ARIMA_Model, VAR_m
from SIRD  import *
from exponential_regression import ExponentialRegression, MultiDimensionalExponentialRegression
from moving_average import MovingAverage, MovingAverageMulti
from Truth import Truth
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from evaluate_model import evaluate_model, evaluate_model_multi, evaluate_model_multi_RMSE, evaluate_model_RMSE
import json

df = pd.read_csv('deaths_and_infections.csv')

# remove a columns from a df: 
df.drop(columns=['Unnamed: 0'], inplace=True)
new_deaths=np.array(df['new_deaths'])
n_infected=np.array(df['n_infected'])
death_cumul=np.array([sum(new_deaths[:i]) for i in range(len(new_deaths))])
dates_of_pandemic=np.arange(len(new_deaths))




# importing mobility from the csv file
df_mobility=pd.read_csv('mobility.csv')
df_mobility.drop(columns=['Unnamed: 0'], inplace=True)
mobility=np.array(df_mobility['mobility'])


df = pd.read_csv('deaths_and_infections.csv')
relier_les_points=[]
for i in range(len(mobility)): 
    if i + 7 < len(mobility): 
        if i % 7 ==0:
            relier_les_points.append(mobility[i])
        else: 
            decalage=i-7*(i//7)
            res = (1-decalage/7)*mobility[7*(i//7)] + (decalage/7)*mobility[7*(i//7)+7]

            relier_les_points.append(res)
    else:
        relier_les_points.append(mobility[i])
mobility_smoothed=np.array(relier_les_points)
data3D=np.array([new_deaths, n_infected, mobility_smoothed])


models1D=['Arima', 'Exponential Regression', 'Moving Average', 'SIRD']
models3D=['Moving Average multi', 'SIRD multi 1', 'SIRD multi 2', 'VAR', 'Exp. Reg. Multi']


for reach in [7, 14]: 

    myarima=ARIMA_Model()
    myexp=ExponentialRegression()
    myexpmulti=MultiDimensionalExponentialRegression()
    mymoving=MovingAverage()
    mysird=SIRD_model_2()
    mysird.choose_model(True, True)
    mysirdmulti1=Multi_SIRD_model()
    mysirdmulti1.choose_model(True, True, True)
    mysirdmulti2=Multi_SIRD_model()
    mysirdmulti2.choose_model(True, True, False)
    myvar=VAR_m()
    mymovingmulti=MovingAverageMulti()
    alphas=np.array([0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    indexs_points=[[5], [10], [15], [20], [25], [30], [35], [40], [45], [50], [55], [60], [65], [70], [75], [80], [85], [90], [95], [100]]
    weights=np.concatenate((np.array([0.5]), alphas * 0.5))
    dicoresults1D=dict()
    dicoresults3D=dict()

    if True: 
        for index_points in indexs_points:
            ############### 1D
            try: 
                perf_arima=evaluate_model_RMSE(model=myarima, data=new_deaths, alphas=alphas, evaluation_point_indexs=index_points, reach=reach, weights=weights)
            except: 
                perf_arima = np.inf
            try: 
                perf_exp=evaluate_model_RMSE(model=myexp, data=new_deaths, alphas=alphas, evaluation_point_indexs=index_points, reach=reach, weights=weights)
            except:
                perf_exp=np.inf
            try: 
                perf_moving=evaluate_model_RMSE(model=mymoving, data=new_deaths, alphas=alphas, evaluation_point_indexs=index_points, reach=reach, weights=weights) 
            except: 
                perf_moving = np.inf
            try : 
                perf_sird=evaluate_model_RMSE(model=mysird, data=new_deaths, alphas=alphas, evaluation_point_indexs=index_points, reach=reach, weights=weights)
            except :
                perf_sird=np.inf
            
            
            

            ### 3D

            try : 
                perfmovingmulti=evaluate_model_multi_RMSE(model=mymovingmulti, data=data3D, alphas=alphas, evaluation_point_indexs=index_points, reach=reach, weights=weights)
            except : 
                perfmovingmulti=np.inf
            try : 
                perf_sirdmulti1=evaluate_model_multi_RMSE(model=mysirdmulti1, data=data3D, alphas=alphas, evaluation_point_indexs=index_points, reach=reach, weights=weights)
            except : 
                perf_sirdmulti1 = np.inf
            try : 
                perf_sirdmulti2=evaluate_model_multi_RMSE(model=mysirdmulti2, data=data3D, alphas=alphas, evaluation_point_indexs=index_points, reach=reach, weights=weights)
            except: 
                perf_sirdmulti2 = np.inf
            try : 
                perfvar=evaluate_model_multi_RMSE(model=myvar, data=data3D, alphas=alphas, evaluation_point_indexs=index_points, reach=reach, weights=weights)
            except : 
                perfvar=np.inf
            try : 
                perfexpmulti=evaluate_model_multi_RMSE(model=myexpmulti, data=data3D, alphas=alphas, evaluation_point_indexs=index_points, reach=reach, weights=weights)
            except: 
                perfexpmulti = np.inf
            
            
            dicoresults1D[str(index_points)]=[perf_arima,perf_exp,  perf_moving, perf_sird]
            dicoresults3D[str(index_points)]=[perfmovingmulti, perf_sirdmulti1, perf_sirdmulti2, perfvar, perfexpmulti]
            

        with open('./results/comparing_models3D_RMSE_reach='+str(reach)+'.json', 'w') as f:
            json.dump(dicoresults3D, f)
        with open('./results/comparing_models1D_RMSE_reach='+str(reach)+'.json', 'w') as f:
                json.dump(dicoresults1D, f)
       