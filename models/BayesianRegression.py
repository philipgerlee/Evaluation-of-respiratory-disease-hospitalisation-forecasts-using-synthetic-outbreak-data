import numpy as np
from Model import Model, Multi_Dimensional_Model
import scipy.stats
from useful_functions import predict_model
from sklearn.linear_model import BayesianRidge

def grad_theta_h_theta(h, theta, x):
    """
    function to compute the gradient of h (which represent the prediction function) with respect to theta, the parameters of the model

    Parameters
    ----------
    h : function
        The function to differenciate
    theta : np.array
        The parameters of the model
    x : np.array
        The input data

    Returns
    -------
    grad : np.array
        The gradient of h with respect to theta

    """

    grad=np.zeros(len(theta))
    for i in range(len(theta)): # this is numerical gradient
        theta_plus=theta.copy()
        theta_plus[i]=theta[i]+0.0001
        grad[i]=(h(theta_plus, x)-h(theta, x))/0.0001
    return grad


def create_br_model(coefs):
    """
    create a Bayesian regression model

    Parameters
    ----------
    coefs : np.array
        The coefficients of the model

    Returns
    -------
    br : BayesianRidge
        The Bayesian Ridge model

    """

    br=BayesianRidge()
    br.fit(np.array([0 for i in range(len(coefs))]).reshape(1, -1), [1])
    br.coef_=coefs
    return br

def prediction_br_model(theta, x ):
    """
    Computes the prediction of the BR model for a given input

    Parameters
    ----------
    theta : np.array
        The parameters of the model
    x : np.array
        The input data

    Returns

    prediction : float
        The prediction of the model

    """
    br = create_br_model(theta)
    return br.predict(x.reshape(1, -1))[0]

class BayesianRegressionModel(Model):


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


        n_days=30
        if n_days >= len(train_dates)/2:
            n_days = len(train_dates)//2 -1 # we avoid to have a RL with more dimensions than the number of points to avoid negative sigma2

        br=BayesianRidge()
        data_ml=[]
        y=[]
        for i in range(n_days, len(data)):
            data_ml.append(data[i-n_days:i])
            y.append(data[i])
        data_ml=np.array(data_ml)
        self.mydata=data_ml
        self.results=br.fit(data_ml, y)
        self.model=br
        self.data=data
        self.trained=True
        self.y=y

    def point_predict(self, reach):
        prediction = predict_model(self.model, self.mydata, self.y, reach)
        last_day=self.mydata[-1]
        last_y=self.y[-1]
        x=np.concatenate((last_day[1:], np.array([last_y]))).reshape(len(last_day), 1)
        self.x=x
        return prediction.reshape(reach,)

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

        prediction = predict_model(self.model, self.mydata, self.y, reach)
        covariance_matrix=self.model.sigma_
        last_day=self.mydata[-1]
        last_y=self.y[-1]
        ci_inf=[]
        ci_up=[]
        x=np.concatenate((last_day[1:], np.array([last_y]))).reshape(len(last_day), 1)
        self.x=x
        self.covariance_matrix=covariance_matrix
        self.grads= []
        self.varps=[]
        for i in range(reach):
            grad=grad_theta_h_theta(prediction_br_model, self.model.coef_, x)
            varp=np.matmul(grad.T, np.matmul(covariance_matrix, grad))
            ci_inf.append(scipy.stats.norm.ppf(alpha/2, loc=prediction[i], scale=np.sqrt(varp))[0])
            ci_up.append(scipy.stats.norm.ppf(1-alpha/2, loc=prediction[i], scale=np.sqrt(varp))[0])
            self.grads.append(grad)
            self.varps.append(varp)
            x=np.concatenate((x[1:], prediction[i].reshape(1, 1)))
        self.grads=np.array(self.grads)
        return prediction.reshape(reach,), list(np.array([ci_inf, ci_up]).reshape(2, reach))
