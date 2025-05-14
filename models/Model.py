import matplotlib.pyplot as plt
import numpy as np

class Model: # base class for all the models

    def __init__(self) :
        self.trained=False
        self.type='1D'
    def reinitialize(self):
        self.trained=False
        self.data=None
        self.train_dates=None
        self.model=None
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
        self.train_dates=train_dates
        self.data=data # to be implemented in the child class
    def predict(self, reach, alphas):
        pass # to be implemented in the child class
    def point_predict(self, reach):
        pass # to be implemented in the child class

    def plot(self, reach, alpha, title=None, xlabel=None, ylabel=None): # to plot the predictions of the model
        '''
        Plot the forecast of the model
        Parameters
        ----------
        reach : int
            The number of periods to forecast
            alpha : float          The confidence level
            title : str            The title of the plot
            xlabel : str           The label of the x-axis
            ylabel : str           The label of the y-axis

        Returns
        -------
        None
        '''
        assert self.trained, 'The model has not been trained yet'
        prediction, intervals = self.predict(reach, alpha)
        ci_low=[max(0, intervals[0][i]) for i in range(len(intervals[0]))]
        ci_high=intervals[1]
        plt.plot([i for i in range(len(self.data))], self.data, label='real data')
        plt.plot([i for i in range(len(self.data), len(self.data) + reach)] , prediction, label='forecast ')
        plt.fill_between([i for i in range(len(self.data), len(self.data) + reach)], ci_low, ci_high, color='black', alpha=.3, label='confidence interval at ' + str(round((1-alpha)*100)) + '%')
        plt.legend()
        plt.axvline(len(self.data), linestyle='--')
        plt.xlim(0,len(self.data)+reach)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        plt.show()



class Multi_Dimensional_Model:
    def __init__(self) :
        self.trained=False
        self.type='3D'
        #Model.plot()
    def reinitialize(self):
        self.trained=False
        self.data=None
        self.train_dates=None
        self.model=None
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
        self.train_dates=train_dates
        self.data=data # to be implemented in the child class
    def predict(self, reach, alphas):
        pass # to be implemented in the child class
    def point_predict(self,reach):
        pass # to be implemented in the child class

    def plot(self, reach, alpha, title=None, xlabel=None, ylabel=None):
        assert self.trained, 'The model has not been trained yet'
        prediction, intervals = self.predict(reach, alpha)
        ci_low=[max(0, intervals[0][i]) for i in range(len(intervals[0]))]
        ci_high=intervals[1]
        plt.plot([i for i in range(len(self.data[0]))], self.data[0], label='real data')
        plt.plot([i for i in range(len(self.data[0]), len(self.data[0]) + reach)] , prediction, label='forecast ')
        plt.fill_between([i for i in range(len(self.data[0]), len(self.data[0]) + reach )], ci_low, ci_high, color='black', alpha=.3, label='confidence interval at ' + str(round((1-alpha)*100)) + '%')
        plt.legend()
        plt.axvline(len(self.data[0]), linestyle='--')
        plt.xlim(0,len(self.data[0])+reach)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()
