import numpy as np
from scipy.stats import wasserstein_distance
from Model import Model
import pandas as pd


def differenciate(x: np.array):
    """
    Compute the approximate differenciation of an array

    Parameters
    ----------
    x : np.array
        The array to differentiate

    Returns
    -------
    np.array
        The approximate derivative of the array

    """
    dx=[x[i+1]-x[i] for i in range(len(x)-1)]
    return dx


def shift(x: np.array, n:int):
    """
    Shift an array by n values
    Parameters
    ----------
    x : np.array
        The array to shift
    n : int


    """
    if n >0 :
        return np.concatenate((np.array([ x[0] for i in range(int(n))]), x))[:len(x)] # we assume that the n first values are the same as the first value of the array
    elif n < 0 :
        return np.concatenate((x, np.array([ x[-1] for i in range(int(-n))])))[-len(x):]
    else :
        return x


def plot_predictions(models: list, data: np.array, dates_of_pandemic: np.array, reach: int, points_of_evaluation: list, fig, ax):
    """
    To plot the predictions of a list of models on the data on which they performed the predictuions

    Parameters
    ----------
    models : list
        The list of models
    data : np.array
        The data on which the models performed the predictions
    dates_of_pandemic : np.array
        The dates of the pandemic
    reach : int
        The number of days to forecast
    points_of_evaluation : list
        The points of evaluation of the models
    fig : matplotlib.figure.Figure
        The figure on which to plot the predictions
    ax : matplotlib.axes.Axes
        The axes on which to plot the predictions

    Returns
    -------
    None

    """
    new_deaths, n_infected, mobility = data
    ax.plot(dates_of_pandemic, new_deaths, c='black')
    colours=['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    for j in range(len(points_of_evaluation)) :
        point = points_of_evaluation[j]
        for i in range(len(models)):
            model=models[i]
            if model.type=='1D':
                model.train( dates_of_pandemic[:point], new_deaths[:point])
                print(model.name)
                try:
                    pred, ints = model.predict(reach, 0.05)
                    if j ==0:
                        ax.plot(np.arange(point, point+reach), pred, c=colours[i], label=model.name)
                    else:
                        ax.plot(np.arange(point, point+reach), pred, c=colours[i] )
                except:
                    print('oups')
            if model.type=='3D':
                model.train(dates_of_pandemic[:point],np.array([new_deaths[:point], n_infected[:point], mobility[:point]]))
                pred, ints=model.predict(7, 0.05)
                if j ==0:
                    ax.plot(np.arange(point, point+reach), pred, c=colours[i], label=model.name)
                else:
                    ax.plot(np.arange(point, point+reach), pred, c=colours[i])
    ax.legend()




def predict_model(model:Model, data_train: np.array,y_train: np.array,  reach: int ):
    """
    Makes the prediction of a model reach days ahead

    Parameters
    ----------
    model : Model
        The model to predict with
    data_train : np.array
        The training data
    y_train : np.array
        The training labels
    reach : int
        The number of days to forecast

    Returns
    -------
    np.array
        The forecasted values

    """
    prediction_reach_days_ahead=[]
    a=np.concatenate((data_train[-1], np.array([y_train[-1]])))[1:]
    predict=model.predict(a.reshape(1, -1))
    prediction_reach_days_ahead.append(predict)
    for i in range(1, reach):
        a=np.concatenate((a[1:], predict))
        predict=model.predict(a.reshape(1, -1))
        prediction_reach_days_ahead.append(predict)
    return np.array(prediction_reach_days_ahead)





def diff_between_2_arrays(array1: np.array, array2: np.array): # this function punishes the pandemics when they do not have the same amplitude (difference in maximum), but also when their derivatives do not have the same amplitude, and when their second derivatives do not have the same amplitude
    """
    Loss to assess the difference between two pandemics to measure the diversity of a set of pandemics

    Parameters
    ----------
    array1 : np.array
        The first pandemic
    array2 : np.array
        The second pandemic

    Returns
    -------
    float
        The diversity measure between the two pandemics

    """
    derive1=np.array(differenciate(array1))
    derive2=np.array(differenciate(array2))
    derivee1=np.array(differenciate(derive1))
    derivee2=np.array(differenciate(derive2))
    max1=max(array1)
    max2=max(array2)
    maxder1=max(derive1)
    maxder2=max(derive2)
    maxderder1=max(derivee1)
    maxderder2=max(derivee2)
    ar1_normalized=array1/np.sum(abs(array1))
    ar2_normalized=array2/np.sum(abs(array2))
    der1_normalized=derive1/np.sum(abs(derive1))
    der2_normalized=derive2/np.sum(abs(derive2))
    derder1_normalized=derivee1/np.sum(abs(derivee1))
    derder2_normalized=derivee2/np.sum(abs(derivee2))
    res=[]
    if max1>max2:
        res.append(max1/max2 -1) # difference of amplitude of the two arrays
    else :
        res.append(max2/max1-1)
    if maxder1>maxder2:
        res.append(maxder1/maxder2-1) # difference of amplitude of the two derivatives
    else :
        res.append(maxder2/maxder1-1)
    if maxderder1>maxderder2:
        res.append(maxderder1/maxderder2-1) # difference of amplitude of the two second derivatives
    else :
        res.append(maxderder2/maxderder1-1)
    res.append(np.sum([abs(ar1_normalized[i]-ar2_normalized[i]) for i in range(len(ar1_normalized))])) # absolute difference of the two arrays
    res.append(np.sum([abs(der1_normalized[i]-der2_normalized[i]) for i in range(len(der1_normalized))])) # absolute difference of the two derivatives
    res.append(np.sum([abs(derder1_normalized[i]-derder2_normalized[i]) for i in range(len(derder1_normalized))])) # absolute difference of the two second derivatives
    res=np.array(res)
    return np.sum(res**2)


def diff_between_2_arrays_2(array1: np.array, array2: np.array): # same function but with wassertsein distance instead of absolute difference

    """
    Loss to assess the difference between two pandemics to measure the diversity of a set of pandemics

    Parameters
    ----------

    array1 : np.array
        The first pandemic
    array2 : np.array
        The second pandemic

    Returns
    -------
    float
        The diversity measure between the two pandemics

    """
    derive1=np.array(differenciate(array1))
    derive2=np.array(differenciate(array2))
    derivee1=np.array(differenciate(derive1))
    derivee2=np.array(differenciate(derive2))
    max1=max(array1)
    max2=max(array2)
    maxder1=max(derive1)
    maxder2=max(derive2)
    maxderder1=max(derivee1)
    maxderder2=max(derivee2)
    ar1_normalized=array1/np.sum(abs(array1))
    ar2_normalized=array2/np.sum(abs(array2))
    der1_normalized=derive1/np.sum(abs(derive1))
    der2_normalized=derive2/np.sum(abs(derive2))
    derder1_normalized=derivee1/np.sum(abs(derivee1))
    derder2_normalized=derivee2/np.sum(abs(derivee2))
    res=[]
    if max1>max2:
        res.append(max1/max2 -1)
    else :
        res.append(max2/max1-1)
    if maxder1>maxder2:
        res.append(maxder1/maxder2-1)
    else :
        res.append(maxder2/maxder1-1)
    if maxderder1>maxderder2:
        res.append(maxderder1/maxderder2-1)
    else :
        res.append(maxderder2/maxderder1-1)
    res.append(wasserstein_distance(ar1_normalized, ar2_normalized))
    res.append(wasserstein_distance(der1_normalized, der2_normalized))
    res.append(wasserstein_distance(derder1_normalized, derder2_normalized))
    res=np.array(res)
    return np.sum(res**2)





def dissemblance_1(pandemic1: np.array, pandemic2: np.array, pandemic3: np.array, pandemic4: np.array):
    """
    assess the diversity of a set of 4 pandemics

    Parameters
    ----------
    pandemic1 : np.array
        The first pandemic
    pandemic2 : np.array
        The second pandemic
    pandemic3 : np.array
        The third pandemic
    pandemic4 : np.array
        The fourth pandemic

    Returns
    -------
    float
        The diversity measure between the four pandemics
    """
    return np.sum([abs(pandemic1[i]-pandemic2[i]) for i in range(len(pandemic1))])+np.sum([abs(pandemic1[i]-pandemic3[i]) for i in range(len(pandemic1))])+np.sum([abs(pandemic1[i]-pandemic4[i]) for i in range(len(pandemic1))])+np.sum([abs(pandemic2[i]-pandemic3[i]) for i in range(len(pandemic1))])+np.sum([abs(pandemic2[i]-pandemic4[i]) for i in range(len(pandemic1))])+np.sum([abs(pandemic3[i]-pandemic4[i]) for i in range(len(pandemic1))])


def dissemblance_2(pandemic1: np.array, pandemic2: np.array, pandemic3: np.array, pandemic4: np.array):
    """
    assess the diversity of a set of 4 pandemics

    Parameters
    ----------
    pandemic1 : np.array
        The first pandemic
    pandemic2 : np.array
        The second pandemic
    pandemic3 : np.array
        The third pandemic
    pandemic4 : np.array
        The fourth pandemic

    Returns
    -------
    float
        The diversity measure between the four pandemics
    """
    return diff_between_2_arrays(pandemic1, pandemic2)+diff_between_2_arrays(pandemic1, pandemic3)+diff_between_2_arrays(pandemic1, pandemic4)+diff_between_2_arrays(pandemic2, pandemic3)+diff_between_2_arrays(pandemic2, pandemic4)+diff_between_2_arrays(pandemic3, pandemic4)


def dissemblance_3(pandemic1: np.array, pandemic2: np.array, pandemic3: np.array, pandemic4: np.array):
    """
    assess the diversity of a set of 4 pandemics

    Parameters
    ----------
    pandemic1 : np.array
        The first pandemic
    pandemic2 : np.array
        The second pandemic
    pandemic3 : np.array
        The third pandemic
    pandemic4 : np.array
        The fourth pandemic

    Returns
    -------
    float
        The diversity measure between the four pandemics
    """
    pandemic1_normalized = np.array(pandemic1/sum(np.abs(pandemic1)))
    pandemic2_normalized = np.array(pandemic2/sum(np.abs(pandemic2)))
    pandemic3_normalized = np.array(pandemic3/sum(np.abs(pandemic3)))
    pandemic4_normalized = np.array(pandemic4/sum(np.abs(pandemic4)))
    return wasserstein_distance(pandemic1_normalized, pandemic2_normalized)+wasserstein_distance(pandemic1_normalized, pandemic3_normalized)+wasserstein_distance(pandemic1_normalized, pandemic4_normalized)+wasserstein_distance(pandemic2_normalized, pandemic3_normalized)+wasserstein_distance(pandemic2_normalized, pandemic4_normalized)+wasserstein_distance(pandemic3_normalized, pandemic4_normalized)


def dissemblance_4(pandemic1: np.array, pandemic2: np.array, pandemic3: np.array, pandemic4: np.array):
    """
    assess the diversity of a set of 4 pandemics

    Parameters
    ----------
    pandemic1 : np.array
        The first pandemic
    pandemic2 : np.array
        The second pandemic
    pandemic3 : np.array
        The third pandemic
    pandemic4 : np.array
        The fourth pandemic

    Returns
    -------
    float
        The diversity measure between the four pandemics
    """
    return diff_between_2_arrays_2(pandemic1, pandemic2)+diff_between_2_arrays_2(pandemic1, pandemic3)+diff_between_2_arrays_2(pandemic1, pandemic4)+diff_between_2_arrays_2(pandemic2, pandemic3)+diff_between_2_arrays_2(pandemic2, pandemic4)+diff_between_2_arrays_2(pandemic3, pandemic4)

def df_to_dict(df: pd.DataFrame):
    df.drop(['Unnamed: 0'], axis=1, inplace=True)
    dict={}
    for column in df.columns :
        dict[column]=list(df[column])
    return dict



def concat_dico(dico1: dict, dico2: dict):
    dico={}
    for key in dico1.keys():
        dico[key]=dico1[key]+dico2[key]
    return dico


def get_classement(maliste: list):
    return [sorted(maliste).index(i) for i in (maliste)]



def sort_list(names: list,index: list):
    names_copy=names.copy()
    index_copy=index.copy()
    names_copy.sort(key=lambda x : index_copy[names.index(x)])
    return names_copy
