from Model import Model, Multi_Dimensional_Model
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.api import VAR

import matplotlib.pyplot as plt
import numpy as np

class ARIMA_Model(Model):

    def train(self, train_dates, data, grid_search=False, p=3, d=0, q=3):
        """
        trains the Arima model on the data

        Parameters
        ----------
        train_dates : list of datetime objects
            The dates of the training data
        data : np.array
            The training data
        grid_search : bool, optional
            If True, the function will try to find the best parameters for the ARIMA model. The default is False.
        p : int, optional
            The number of autoregressive terms. The default is 3.
        d : int, optional
            The order of differenciation. The default is 0.
        q : int, optional
            The number of moving average terms. The default is 3.

        Returns
        -------
        None



        """
        self.name='Arima'
        self.data=data
        if grid_search:
            min = 1000
            pmin=0
            dmin=0
            qmin=0
            for p in range(7):
                for d in range(4):
                    for q in range(7):
                        if not (p,d,q) == (0,1,0):
                            model = ARIMA(data, order=(p,d,q))
                            try :
                                fitted = model.fit()
                                prediction=fitted.predict(start=1, end=len(data) , typ='levels')
                                error = np.mean(np.abs(prediction - data))
                                if error < min:
                                    min = error
                                    pmin=p
                                    dmin=d
                                    qmin=q
                            except np.linalg.LinAlgError as err :
                                pass
        else :
            pmin=p
            dmin=d
            qmin=q
        self.model = ARIMA(data, order=(pmin,dmin,qmin))
        self.p=pmin
        self.d=dmin
        self.q=qmin
        self.fitted=self.model.fit()
        self.trained= True

    def point_predict(self, reach):
        assert self.trained, 'The model has not been trained yet'
        predifore=self.fitted.get_forecast(reach).predicted_mean # this is an array of size reach
        assert type(predifore) == np.ndarray, 'The prediction should be a numpy array'
        for i in range(len(predifore)):
            if predifore[i] < 0:
                predifore[i]=0
        return predifore


    def predict(self, reach, alpha):
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
        predifore=self.fitted.get_forecast(reach).predicted_mean # this is an array of size reach
        assert type(predifore) == np.ndarray, 'The prediction should be a numpy array'
        interval = self.fitted.get_forecast(reach).conf_int(alpha=alpha)
        ci_low=[max(elt[0],0) for elt in interval]
        ci_high=[elt[1] for elt in interval]
        for i in range(len(predifore)):
            if predifore[i] < 0:
                predifore[i]=0
        return predifore, [ci_low, ci_high]




class VAR_m(Multi_Dimensional_Model):

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
        self.name='VAR'
        # the data is a array of shape (3, n) where n is the number of days
        # we have to transpose it so the VAR model reads it correctly but not in the self.data attribute as it is used in a different way for the .plot() method in the mother class
        self.data=data
        self.model=VAR(self.data.transpose())
        self.fitted=self.model.fit(trend='n')
        self.trained= True

    def point_predict(self,reach):
        assert self.trained, 'The model has not been trained yet'
        lag=self.fitted.k_ar
        ints=self.fitted.forecast_interval(self.data.transpose()[-lag:], steps=reach, alpha=0.05)
        pred=ints[0].transpose()[0]
        return pred

    def predict(self, reach, alpha):

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
        lag=self.fitted.k_ar
        ints=self.fitted.forecast_interval(self.data.transpose()[-lag:], steps=reach, alpha=alpha)
        pred=ints[0].transpose()[0]
        low=ints[1].transpose()[0]
        high=ints[2].transpose()[0]
        low[np.where(low == None)] = np.nan
        high[np.where(high == None)] = np.nan
        if np.isnan(low).any(): # we adapt the ci in the case of nan values
            low=[0 for i in range(len(low))]
        if np.isnan(high).any(): # we adapt the ci in the case of nan values
            high=[abs(1000*max(pred)) for i in range(len(high))]
        return pred, [list(low), list(high)]
