import numpy as np
from Model import Model



def IS(interval : tuple, point : float, alpha: float) -> float:
    """
    Compute the interval score of a prediction
    Parameters
    ----------
    interval : tuple
        The confidence interval
    point : float
        The point to evaluate the prediction on
    alpha : float
        The confidence level

    Returns
    -------
    float
        The interval score of the prediction


    """
    #assert interval[0] <= interval[1] , print(interval[0], interval[1])

    assert alpha >= 0
    assert alpha <= 1
    l=interval[0]
    u = interval[1]
    dispersion = u-l

    if point < l :

        underprediction= (2/alpha)*(l-point)
    else:
        underprediction=0
    if point > u :

        overprediction = (2/alpha)*(point-u)
    else:
        overprediction=0

    return underprediction + overprediction + dispersion



def WIS(prediction: float, intervals : list, point_of_evaluation : float, alphas: list, weights: list) -> float:
    """
    point of evaluation  is the real value that we try to predict

    WIS computes the Weighted Interval Score of a model that predicts a point and a list of confidence intervals.
    The fuction taks as an input a prediction, a list of confidence intervals of precision alpha, a list of weights to apply to the different intervals and a point to evaluate the prediction on.

    """
    assert all([alpha >= 0 for alpha in alphas])
    assert all([alpha <= 1 for alpha in alphas])
    assert len(alphas)==len(weights)-1
    K = len(alphas)
    loss=0
    for k in range( K):
        alpha=alphas[k]
        interval=intervals[k]
        loss += weights[k+1]*IS(interval, point_of_evaluation, alpha) # the first weight is for the prediction
    loss += weights[0]* abs(prediction - point_of_evaluation)
    return loss/(K+1/2)


def evaluate_model(model: Model, data: np.array, alphas: list, evaluation_point_indexs: list, reach: int, weights: list) -> float:

    loss=0
    for index in evaluation_point_indexs:
        model.train(train_dates = [i for i in range(index)], data = data[:index] )
        intervals=[]
        for alpha in alphas:
            prediction, interval = model.predict(reach, alpha)
            interval_low=interval[0][-1]
            interval_high=interval[1][-1]
            intervals.append((interval_low, interval_high))
        prediction=prediction[-1]
        wis=WIS(prediction=prediction, intervals = intervals, point_of_evaluation = data[index+reach-1], alphas = alphas , weights = weights)
        loss+=wis
    return loss / len(evaluation_point_indexs) # average loss over all evaluation points



def evaluate_model_multi(model: Model, data: np.array, alphas: list, evaluation_point_indexs: list, reach: int, weights: list) -> float:
    loss=0
    for index in evaluation_point_indexs:
        data_train=data.transpose()[:index].transpose()
        model.train(train_dates = [i for i in range(index)], data = data_train )
        intervals=[]
        for alpha in alphas:
            prediction, interval = model.predict(reach, alpha)
            interval_low=interval[0][-1]
            interval_high=interval[1][-1]
            intervals.append((interval_low, interval_high))
        prediction=prediction[-1]
        wis=WIS(prediction=prediction, intervals = intervals, point_of_evaluation = data[0][index+reach-1], alphas = alphas , weights = weights)
        loss+=wis
    return loss / len(evaluation_point_indexs) # average loss over all evaluation points


def evaluate_model_RMSE(model: Model, data: np.array, alphas: list, evaluation_point_indexs: list, reach: int, weights: list) -> float:
    loss=0
    for index in evaluation_point_indexs:
        model.train(train_dates = [i for i in range(index)], data = data[:index] )
        intervals=[]
        for alpha in alphas:
            prediction, interval = model.predict(reach, alpha)
            interval_low=interval[0][-1]
            interval_high=interval[1][-1]
            intervals.append((interval_low, interval_high))
        prediction=prediction[-1]
        RMSE=np.sqrt((prediction - data[index+reach-1])**2)
        loss+=RMSE
    return loss / len(evaluation_point_indexs) # average loss over all evaluation points




def evaluate_model_multi_RMSE(model: Model, data: np.array, alphas: list, evaluation_point_indexs: list, reach: int, weights: list) -> float:
    loss=0
    for index in evaluation_point_indexs:
        data_train=data.transpose()[:index].transpose()
        model.train(train_dates = [i for i in range(index)], data = data_train )
        intervals=[]
        for alpha in alphas:
            prediction, interval = model.predict(reach, alpha)
            interval_low=interval[0][-1]
            interval_high=interval[1][-1]
            intervals.append((interval_low, interval_high))
        prediction=prediction[-1]
        RMSE=np.sqrt((prediction - data[0][index+reach-1])**2)
        loss+=RMSE
    return loss / len(evaluation_point_indexs) # average loss over all evaluation points
