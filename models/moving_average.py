import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
from Model import Model, Multi_Dimensional_Model

# baselines models to be compared with the others

class MovingAverage(Model):
    def train(self, train_dates, data):
        """

        Trains the model on the data

        Parameters
        ----------
        train_dates : list of datetime objects
            The dates of the training data
        data : np.array
            The training data

        Returns
        -------
        None



        """
        self.data=data
        self.name = 'Moving Average'
        self.train_dates=train_dates
        self.value=np.mean(data[-7:])
        self.trained=True

    def point_predict(self, reach):
        assert self.trained, 'The model has not been trained yet'
        self.prediction=np.array([self.value for i in range(reach)])
        return self.prediction

    def predict(self, reach, alpha, method='hessian'):

        """
        Predicts the number of cases for the next reach days

        Parameters
        ----------
        reach : int
            The number of days to forecast
        alpha : float
            The confidence level

        Returns
        -------
        predifore : np.array
            The forecasted number of cases
        [ci_low, ci_high] : list of np.array


        """



        assert self.trained, 'The model has not been trained yet'
        sigma2=(1/(7-1))*np.sum((self.data[-7:]-self.value)**2)
        self.prediction=np.array([self.value for i in range(reach)])
        m_sampled=[]
        intervals=[self.prediction]
        for i in range(100):

            m_r=self.value+np.random.normal(0, np.sqrt(sigma2), 1)[0]
            m_sampled.append(m_r)
            prediction_sampled=[m_r for i in range(reach)]
            intervals.append(prediction_sampled)
        self.m_sampled=m_sampled
        intervals=np.array(intervals).transpose()
        self.intervals=intervals
        ci_low=np.array([np.quantile(intervals[i], alpha/2) for i in range(reach)])
        ci_high=np.array([np.quantile(intervals[i],1-alpha/2) for i in range(reach)])
        return self.prediction, [ci_low, ci_high]



class MovingAverageMulti(Multi_Dimensional_Model):
    def train(self, train_dates, data):
        """

        Trains the model on the data

        Parameters
        ----------
        train_dates : list of datetime objects
            The dates of the training data
        data : np.array
            The training data

        Returns
        -------
        None



        """
        self.name = 'Moving Average multi'
        self.data=data
        self.train_dates=train_dates
        self.value=np.mean(data[0][-7:])
        self.trained=True

    def predict(self, reach, alpha, method='hessian'):
        assert self.trained, 'The model has not been trained yet'
        sigma2=(1/(7-1))*np.sum((self.data[0][-7:]-self.value)**2)
        self.prediction=np.array([self.value for i in range(reach)])
        m_sampled=[]
        intervals=[self.prediction]
        for i in range(100):

            m_r=self.value+np.random.normal(0, np.sqrt(sigma2), 1)[0]
            m_sampled.append(m_r)
            prediction_sampled=[m_r for i in range(reach)]
            intervals.append(prediction_sampled)
        self.m_sampled=m_sampled
        intervals=np.array(intervals).transpose()
        self.intervals=intervals
        ci_low=np.array([np.quantile(intervals[i], alpha/2) for i in range(reach)])
        ci_high=np.array([np.quantile(intervals[i],1-alpha/2) for i in range(reach)])
        return self.prediction, [ci_low, ci_high]
