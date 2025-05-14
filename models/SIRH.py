from Model import Model, Multi_Dimensional_Model
from scipy.optimize import curve_fit, minimize
import numpy as np
from scipy.optimize import differential_evolution
from scipy.integrate import odeint, ode
from scipy.interpolate import interp1d

import pandas as pd
import sys
sys.path.append('./../')
import scipy.stats

const_gamma_i=1/8
const_gamma_h=1/18

rho=1/5
gamma=1/8
lag=21


s_0=1000000 -1
e_0=0
i_0=1
r_0=0
h_0=0
dt=0.001


def derive_sirh(x, beta, N, gamma_i,gamma_h,  h): # the derivative of the sirh model to be used in the run_sirh function
    S=x[0]
    I=x[1]
    R=x[2]
    H=x[3]
    return np.array([-beta*S*I/N, beta*S*I/N - (gamma_i+h)*I  , gamma_i*I + gamma_h *H, h * I - gamma_h * H ])

def sirh_rhs(x,t,beta,N,gamma_i,gamma_h,h):
    S=x[0]
    I=x[1]
    R=x[2]
    H=x[3]
    return np.array([-beta*S*I/N, beta*S*I/N - (gamma_i+h)*I  , gamma_i*I + gamma_h *H, h * I - gamma_h * H ])

def run_sirh(x0, beta, gamma_i, gamma_h,h,  t, dt):
    T=np.arange(t)
    N=np.sum(x0)
    solution=odeint(sirh_rhs,x0,T,args=(beta,N,gamma_i,gamma_h,h))
    # Unpack the solution array into separate state variables
    S, I, R, H = solution.T  # Transpose and unpack
    return S, I, R, H

def run_sirh_old(x0, beta, gamma_i, gamma_h,h,  t, dt):
    """
    Runs a Euler intergation of the equation of the SIRH model

    Parameters
    ----------
    x0 : np.array
        The initial state of the model
    beta : float
        The transmission rate
    gamma_i : float
        The recovery rate of the infected individuals
    gamma_h : float
        The recovery rate of the hospitalized individuals
    h : float
        The rate of hospitalization
    t : float
        The number of days to forecast
    dt : float
        The time step of the integration

    Returns
    -------
    s_final : np.array
        The time-series of susceptible individuals
    i_final : np.array
        The time-serie of infected individuals
    r_final : np.array
        The time-serie of recovered individuals
    h_final : np.array
        The time-serie of hospitalized individuals
    """

    x=x0
    S=[x[0]]
    I=[x[1]]
    R=[x[2]]
    H=[x[3]] # hospitalized
    n_iter=int(t/dt)
    N=sum(x0)
    for i in range(n_iter):
        x=x+dt*derive_sirh(x, beta, N, gamma_i, gamma_h, h)
        S.append(x[0])
        I.append(x[1])
        R.append(x[2])
        H.append(x[3])
    s_final=[]
    i_final=[]
    r_final=[]
    h_final=[]
    time=np.linspace(0, t, int(t/dt) )
    for i in range(len(time)-1):
        if abs(time[i]-int(time[i]))<dt:
            s_final.append(S[i])
            i_final.append(I[i])
            r_final.append(R[i])
            h_final.append(H[i])
    s_final.append(S[-1])
    i_final.append(I[-1])
    r_final.append(R[-1])
    h_final.append(H[-1])
    s_final=s_final[1:]
    i_final=i_final[1:]
    r_final=r_final[1:]
    h_final=h_final[1:]
    return s_final, i_final, r_final, h_final




def differenciate(x):
    dx=[x[i+1]-x[i] for i in range(len(x)-1)]
    return dx

def sirh_for_optim(x, beta, gamma_i, gamma_h, h):
    """
    Returns the number of hospitalized individuals for each day of the pandemic for the optimization function for SIRH1
    """
    # x is a list of dates (0 - 122)
    x0=[s_0, i_0, r_0, h_0]
    t=len(x)
    S,I,R,H=run_sirh(x0, beta, gamma_i, gamma_h,h,  t, dt)
    h_arr=np.array(H)
    return h_arr





def sirh_for_optim_3(x, beta,gamma_h,  h):
    """
    Returns the number of hospitalized individuals for each day of the pandemic for the optimization function for SIRH3
    """
    # x is a list of dates (0 - 122)
    x0=[s_0, i_0, r_0, h_0]
    t=len(x)
    S,I,R,H=run_sirh(x0, beta, const_gamma_i,gamma_h, h,  t, dt)
    h_arr=np.array(H)
    return h_arr


def sirh_for_optim_4(x, beta, gamma_i,  h):
    """
    Returns the number of hospitalized individuals for each day of the pandemic for the optimization function for SIRH4
    """
    # x is a list of dates (0 - 122)
    x0=[s_0, i_0, r_0, h_0]
    t=len(x)
    S,I,R,H=run_sirh(x0, beta, gamma_i, const_gamma_h, h,  t, dt)
    h_arr=np.array(H)
    return h_arr

def sirh_for_optim_2(x, beta, h):
    """
    Returns the number of hospitalized individuals for each day of the pandemic for the optimization function for SIRH2
    """
    # x is a list of dates (0 - 122)
    x0=[s_0, i_0, r_0, h_0]
    t=len(x)
    S,I,R,H=run_sirh(x0, beta, const_gamma_i, const_gamma_h, h,  t, dt)
    h_arr=np.array(H)
    return h_arr

def grad_theta_h_theta(x0, theta, reach, the_gamma_constant=None): # different case that depend on the number of parameters to optimize
    """
    Compute the gradient of the estimation with respect to theta

    Parameters
    ----------
    x0 : np.array
        The initial state of the model
    theta : np.array
        The parameters of the model
    reach : int
        The number of days to forecast
    the_gamma_constant : str
        The gamma constant that is not optimized

    Returns
    -------
    grad : np.array
        The gradient of the estimation with respect to theta

    """
    grad=np.zeros((len(theta), reach))
    for i in range(len(grad)):
        if len(theta)==2:
            theta_plus=theta.copy()
            theta_plus[i]+=0.0001
            _, _, _, hospitalized_grad = run_sirh([x0[0], x0[1], x0[2], x0[3]], theta_plus[0], const_gamma_i,const_gamma_h,  theta_plus[1], reach, 0.001)
            _, _, _, hospitalized = run_sirh([x0[0], x0[1], x0[2], x0[3]], theta[0], const_gamma_i, const_gamma_h, theta[1], reach, 0.001)
        elif len(theta)==4:
            theta_plus=theta.copy()
            theta_plus[i]+=0.0001
            _, _, _, hospitalized_grad = run_sirh([x0[0], x0[1], x0[2], x0[3]], theta_plus[0],  theta_plus[1],theta_plus[2], theta_plus[3],  reach, 0.001)
            _, _, _, hospitalized = run_sirh([x0[0], x0[1], x0[2], x0[3]], theta[0], theta[1], theta[2],theta_plus[3],  reach, 0.001)
        elif len(theta)==3:
            theta_plus=theta.copy()
            theta_plus[i]+=0.0001
            if the_gamma_constant=='gamma_h':
                _, _, _, hospitalized_grad = run_sirh([x0[0], x0[1], x0[2], x0[3]], theta_plus[0],  theta_plus[1],const_gamma_h,  theta_plus[2],  reach, 0.001)
                _, _, _, hospitalized = run_sirh([x0[0], x0[1], x0[2], x0[3]], theta[0], theta[1], const_gamma_h, theta[2],  reach, 0.001)
            elif the_gamma_constant=='gamma_i':
                _, _, _, hospitalized_grad = run_sirh([x0[0], x0[1], x0[2], x0[3]], theta_plus[0],  const_gamma_i,theta_plus[1],  theta_plus[2],  reach, 0.001)
                _, _, _, hospitalized = run_sirh([x0[0], x0[1], x0[2], x0[3]], theta[0],  const_gamma_i, theta[1], theta[2],  reach, 0.001)


        h_arr_grad=(np.array(hospitalized_grad))
        h_arr=(np.array(hospitalized))
        grad[i]=(h_arr_grad-h_arr)/0.0001
    return grad



class SIRH_model_2(Model):
    """
    A SIRH model of the second type that uses the value of the mobility to get a time varying transmission rate and is fitted to both number of hospitalized and iinfected curves
    """
    s_0=1000000 -1
    i_0=1
    r_0=0
    h_0=0
    dt=0.001
    def choose_model(self, gamma_i_constant,gamma_h_constant,  reset_state):
        self.gamma_i_constant=gamma_i_constant
        self.gamma_h_constant=gamma_h_constant
        self.reset_state=reset_state
        self.N=s_0+i_0+r_0+h_0
    def train( self, train_dates,  data):
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
        self.name='SIRH'
        self.data=data
        self.train_dates=train_dates
        if self.gamma_i_constant and self.gamma_h_constant:
            p,cov= curve_fit(sirh_for_optim_2, [i for i in range(len(data))],data, p0=[ 5.477e-01  , 5.523e-04],  bounds=([0,0], [np.inf,np.inf]),ftol=1e-3, xtol=1e-3)
            self.beta=p[0]
            self.h=p[1]
            self.gamma_i=const_gamma_i
            self.gamma_h=const_gamma_h
            self.cov=cov
            self.trained= True
        elif not self.gamma_i_constant and not self.gamma_h_constant:
            p,cov= curve_fit(sirh_for_optim, self.train_dates,data, p0=[ 5.477e-01 , 2.555e-02 ,2.555e-02,  5.523e-04],  bounds=([0,0,0, 0], [10,10, 10, 10]),ftol=1e-3, xtol=1e-3)
            self.beta=p[0]
            self.gamma_i=p[1]
            self.gamma_h=p[2]
            self.h=p[3]
            self.cov=cov
            self.trained= True
        elif self.gamma_i_constant and not self.gamma_h_constant:
            p,cov= curve_fit(sirh_for_optim_3, self.train_dates,data, p0=[ 5.477e-01 , 2.555e-02 ,  5.523e-04],  bounds=([0,0, 0], [10,10, 10]),ftol=1e-3, xtol=1e-3)
            self.beta=p[0]
            self.gamma_i=const_gamma_i
            self.gamma_h=p[1]
            self.h=p[2]
            self.cov=cov
            self.trained= True
        else:
            p,cov= curve_fit(sirh_for_optim_4, self.train_dates,data, p0=[ 5.477e-01 , 2.555e-02 ,  5.523e-04],  bounds=([0,0, 0], [10,10, 10]),ftol=1e-3, xtol=1e-3)
            self.beta=p[0]
            self.gamma_i=p[1]
            self.gamma_h=const_gamma_h
            self.h=p[2]
            self.cov=cov
            self.trained= True
        #print(p)

    def point_predict(self,reach):
        S,I,R,H=run_sirh([s_0, i_0, r_0, h_0], self.beta, self.gamma_i, self.gamma_h, self.h , len(self.train_dates), 0.001)
        self.S=S
        self.I=I
        self.R=R
        self.H=H
        assert self.trained, 'The model has not been trained yet'
        if self.reset_state:
            s_t=S[-1]
            i_t=I[-1]
            h_t=self.data[-1]
            r_t=self.N-s_t-i_t-h_t
            _, _, _, hospitalized=run_sirh([s_t, i_t, r_t, h_t], self.beta, self.gamma_i, self.gamma_h, self.h, reach, 0.001)
        else :
            #hospitalized=sirh_for_optim(np.array([i for i in range(len(self.train_dates)+reach)]), self.beta, self.gamma_i, self.gamma_h,self.h)
            s_t=S[-1]
            i_t=I[-1]
            h_t=H[-1]
            r_t=R[-1]
            _, _, _, hospitalized=run_sirh([s_t, i_t, r_t, h_t], self.beta, self.gamma_i, self.gamma_h, self.h, reach, 0.001)
            offset=H[-1]-self.data[-1]
            hospitalized=hospitalized-offset
        self.prediction =  hospitalized
        prediction=self.prediction
        return prediction

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


        S,I,R,H=run_sirh([s_0, i_0, r_0, h_0], self.beta, self.gamma_i, self.gamma_h, self.h , len(self.train_dates), 0.001)
        self.S=S
        self.I=I
        self.R=R
        self.H=H
        assert self.trained, 'The model has not been trained yet'
        if self.reset_state:
            s_t=S[-1]
            i_t=I[-1]
            h_t=self.data[-1]
            r_t=self.N-s_t-i_t-h_t
            _, _, _, hospitalized=run_sirh([s_t, i_t, r_t, h_t], self.beta, self.gamma_i, self.gamma_h, self.h, reach, 0.001)
        else :
            #hospitalized=sirh_for_optim(np.array([i for i in range(len(self.train_dates)+reach)]), self.beta, self.gamma_i, self.gamma_h,self.h)
            s_t=S[-1]
            i_t=I[-1]
            h_t=H[-1]
            r_t=R[-1]
            _, _, _, hospitalized=run_sirh([s_t, i_t, r_t, h_t], self.beta, self.gamma_i, self.gamma_h, self.h, reach, 0.001)
            offset=H[-1]-self.data[-1]
            hospitalized=hospitalized-offset
        self.prediction =  hospitalized
        prediction=self.prediction
        ci_low=[]
        ci_high=[]
        if self.gamma_i_constant and self.gamma_h_constant:
            grad=grad_theta_h_theta([self.S[-1], self.I[-1], self.R[-1], self.H[-1]], [self.beta, self.h], reach) # size 2 x reach
        elif not self.gamma_i_constant and not self.gamma_h_constant:
            grad=grad_theta_h_theta([self.S[-1], self.I[-1], self.R[-1], self.H[-1]], [self.beta,self.gamma_i,self.gamma_h,   self.h], reach) # size 4 x reach
        elif self.gamma_i_constant and not self.gamma_h_constant:
            grad=grad_theta_h_theta([self.S[-1], self.I[-1], self.R[-1], self.H[-1]], [self.beta,self.gamma_h,  self.h], reach, the_gamma_constant='gamma_i')
        else:
            grad=grad_theta_h_theta([self.S[-1], self.I[-1], self.R[-1], self.H[-1]], [self.beta,self.gamma_i,  self.h], reach, the_gamma_constant='gamma_h')
        cov=self.cov
        vars=np.diagonal((grad.transpose() @ cov @ grad).transpose())
        self.vars=vars
        self.grad=grad
        assert(len(vars)==reach, str(len(vars)) + 'different from ' + str(reach))
        for i in range(len(vars)):
            down = scipy.stats.norm.ppf(alpha/2, loc=self.prediction[i], scale=np.sqrt(vars[i]))
            ci_low.append(down)
            up = scipy.stats.norm.ppf(1-(alpha/2), loc=self.prediction[i], scale=np.sqrt(vars[i]))
            ci_high.append(up)
        self.ci_low=ci_low
        self.ci_high=ci_high
        return prediction, [ci_low, ci_high]

    def predict_variance(self, reach):
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


        S,I,R,H=run_sirh([s_0, i_0, r_0, h_0], self.beta, self.gamma_i, self.gamma_h, self.h , len(self.train_dates), 0.001)
        self.S=S
        self.I=I
        self.R=R
        self.H=H
        assert self.trained, 'The model has not been trained yet'
        if self.reset_state:
            s_t=S[-1]
            i_t=I[-1]
            h_t=self.data[-1]
            r_t=self.N-s_t-i_t-h_t
            _, _, _, hospitalized=run_sirh([s_t, i_t, r_t, h_t], self.beta, self.gamma_i, self.gamma_h, self.h, reach, 0.001)
        else :
            #hospitalized=sirh_for_optim(np.array([i for i in range(len(self.train_dates)+reach)]), self.beta, self.gamma_i, self.gamma_h,self.h)
            s_t=S[-1]
            i_t=I[-1]
            h_t=H[-1]
            r_t=R[-1]
            _, _, _, hospitalized=run_sirh([s_t, i_t, r_t, h_t], self.beta, self.gamma_i, self.gamma_h, self.h, reach, 0.001)
            offset=H[-1]-self.data[-1]
            hospitalized=hospitalized-offset
        self.prediction =  hospitalized
        prediction=self.prediction
        ci_low=[]
        ci_high=[]
        if self.gamma_i_constant and self.gamma_h_constant:
            grad=grad_theta_h_theta([self.S[-1], self.I[-1], self.R[-1], self.H[-1]], [self.beta, self.h], reach) # size 2 x reach
        elif not self.gamma_i_constant and not self.gamma_h_constant:
            grad=grad_theta_h_theta([self.S[-1], self.I[-1], self.R[-1], self.H[-1]], [self.beta,self.gamma_i,self.gamma_h,   self.h], reach) # size 4 x reach
        elif self.gamma_i_constant and not self.gamma_h_constant:
            grad=grad_theta_h_theta([self.S[-1], self.I[-1], self.R[-1], self.H[-1]], [self.beta,self.gamma_h,  self.h], reach, the_gamma_constant='gamma_i')
        else:
            grad=grad_theta_h_theta([self.S[-1], self.I[-1], self.R[-1], self.H[-1]], [self.beta,self.gamma_i,  self.h], reach, the_gamma_constant='gamma_h')
        cov=self.cov
        vars=np.diagonal((grad.transpose() @ cov @ grad).transpose())
        self.vars=vars
        self.grad=grad
        return prediction, vars

def sirh_m_rhs(x,t,a,b,N,gamma_i,gamma_h,h,mob):
    S=x[0]
    I=x[1]
    R=x[2]
    H=x[3]
    return np.array([-(a*mob(t)+b)*S*I/N, (a*mob(t)+b)*S*I/N - (gamma_i+h)*I  , gamma_i*I + gamma_h *H, h * I - gamma_h * H ])


def run_sirh_m(x0, a, b, gamma_i, gamma_h,h, mob, t0,t1):
    T=np.arange(t0,t1)
    N=np.sum(x0)
    #mob=interp1d(np.arange(len(mobility)),mobility,kind="linear",fill_value="extrapolate")
    solution=odeint(sirh_m_rhs,x0,T,args=(a,b,N,gamma_i,gamma_h,h,mob))
    # Unpack the solution array into separate state variables
    S, I, R, H = solution.T  # Transpose and unpack
    return S, I, R, H



def run_sirh_m_old(x0, a, b , gamma_i, gamma_h,h, mobility , dt):
    """
    Runs a Euler intergation of the equation of the multi SIRH model

    Parameters
    ----------
    x0 : np.array
        The initial state of the model
    a : float
        The intercept of the transmission rate
    b : float
        The slope of the transmission rate
    gamma_i : float
        The recovery rate of the infected individuals
    gamma_h : float
        The recovery rate of the hospitalized individuals
    h : float
        The rate of hospitalization
    mobility : np.array
        The mobility data
    dt : float
        The time step of the integration

    Returns
    -------
    s_final : np.array
        The time-series of susceptible individuals
    i_final : np.array
        The time-serie of infected individuals
    r_final : np.array
        The time-serie of recovered individuals
    h_final : np.array
        The time-serie of hospitalized individuals
    """
    t=len(mobility)
    x=x0
    S=[x[0]]
    I=[x[1]]
    R=[x[2]]
    H=[x[3]] # hospitalized
    n_iter=int(t/dt)
    N=sum(x0)
    for i in range(n_iter):
        todays_mobility=mobility[int(i*dt)]
        beta=a*todays_mobility+b
        x=x+dt*derive_sirh(x, beta, N, gamma_i, gamma_h, h)
        S.append(x[0])
        I.append(x[1])
        R.append(x[2])
        H.append(x[3])
    s_final=[]
    i_final=[]
    r_final=[]
    h_final=[]
    time=np.linspace(0, t, int(t/dt) )
    for i in range(len(time)-1):
        if abs(time[i]-int(time[i]))<dt:
            s_final.append(S[i])
            i_final.append(I[i])
            r_final.append(R[i])
            h_final.append(H[i])
    return s_final, i_final, r_final, h_final

def sirh_for_optim_m_old(x, a, b ,h, mobility, gamma_i=const_gamma_i, gamma_h=const_gamma_h): # returns first the number of deaths and then the number of total infected
    """
    Concatenation of both infected and hospitalized individuals for the optimization function

    Parameters
    ----------
    x : np.array
        The dates of the pandemic
    a : float
        The intercept of the transmission rate
    b : float
        The slope of the transmission rate
    h : float
        The rate of hospitalization
    mobility : np.array
        The mobility data
    gamma_i : float
        The recovery rate of the infected individuals
    gamma_h : float
        The recovery rate of the hospitalized individuals
    T  : int
        The number of days

    Returns
    -------
    np.array
        The concatenation of the hospitalized and infected individuals

    """
    s_0=1000000 -1
    i_0=1
    r_0=0
    h_0=0
    x0=np.array([s_0, i_0, r_0, h_0])
    dt=0.001

    S, I, R, H = run_sirh_m(x0, a, b , gamma_i, gamma_h, h, mobility , 0,len(x))
    h_arr=np.array(H)
    I_arr=np.array(I)
    return np.concatenate((h_arr, I_arr))

def sirh_for_optim_m(x,a,b,h,mob,gamma_i=const_gamma_i, gamma_h=const_gamma_h):
    x0=[s_0, i_0, r_0, h_0]
    t=len(x)
    S,I,R,H=run_sirh_m(x0,a,b,gamma_i, gamma_h,h,mob,0,t)
    h_arr=np.array(H)
    return h_arr


def sirh_for_optim_normalized(x, a, b, h, mobility, n_hospitalized, n_infected,  taking_I_into_account=True, gamma_i=const_gamma_i, gamma_h=const_gamma_h): # returns firts the number of deaths and then the number of total infected
    """
    Normalization of the number of hospitalized and infected individuals for the optimization function

    Parameters
    ----------
    x : np.array
        The dates of the pandemic
    a : float
        The intercept of the transmission rate
    b : float
        The slope of the transmission rate
    h : float
        The rate of hospitalization
    mobility : np.array
        The mobility data
    n_hospitalized : np.array
        The time series of the number of hospitalized individuals
    n_infected : np.array
        The time series of the number of infected individuals
    taking_I_into_account : bool
        Whether to take the infected individuals into account or not
    gamma_i : float
        The recovery rate of the infected individuals
    gamma_h : float
        The recovery rate of the hospitalized individuals

    Returns
    -------
    np.array
       The objective function to fit in the training phase

    """
    I_and_H=sirh_for_optim_m(x, a, b, h, mobility, gamma_i, gamma_h)
    I=I_and_H[len(I_and_H)//2:]
    H=I_and_H[:len(I_and_H)//2]
    if taking_I_into_account:
        return np.concatenate((H/np.max(n_hospitalized), I/np.max(n_infected)))
    else:
        return H



def grad_theta_h_theta_m(x0, theta, mob,start,end): # for gamma constant
    """
    compute the gradient of the estimation with respect to theta

    Parameters
    ----------
    x0 : np.array
        The initial state of the model
    theta : np.array
        The parameters of the model
    mob_predicted : np.array
        The mobility data estimated that will be used in the prediction
    """
    reach=end-start
    grad=np.zeros((len(theta), reach))
    for i in range(len(grad)):
        theta_plus=theta.copy()
        theta_plus[i]+=0.0001
        _, _, _, hospitalized_grad = run_sirh_m([x0[0], x0[1], x0[2], x0[3]], theta_plus[0], theta_plus[1], const_gamma_i,const_gamma_h, theta_plus[2] , mob, start,end)
        _, _, _, hospitalized= run_sirh_m([x0[0], x0[1], x0[2], x0[3]], theta[0], theta[1], const_gamma_i,const_gamma_h,  theta[2], mob, start,end)
        hospitalized_arr_grad= (np.array(hospitalized_grad))
        hospitalized_arr=(np.array(hospitalized))
        grad[i]=(hospitalized_arr_grad-hospitalized_arr)/0.0001
    return grad




def grad_theta_h_theta_m_bis(x0, theta, mob,start,end): # for gamma not constants
    """
    Compute the gradient of the estimation with respect to theta

    Parameters
    ----------
    x0 : np.array
        The initial state of the model
    theta : np.array
        The parameters of the model
    mob_predicted : np.array
        The mobility data estimated that will be used in the prediction

    Returns
    -------
    grad : np.array
        The gradient of the estimation with respect to theta

    """
    reach=end-start
    grad=np.zeros((len(theta), reach))
    for i in range(len(grad)):
        theta_plus=theta.copy()
        theta_plus[i]+=0.0001
        _, _, _, hospitalized_grad = run_sirh_m([x0[0], x0[1], x0[2], x0[3]], theta_plus[0], theta_plus[1],  theta_plus[2],  theta_plus[3],  theta_plus[4] , mob,start,end)
        _, _, _, hospitalized= run_sirh_m([x0[0], x0[1], x0[2], x0[3]], theta[0], theta[1],  theta[2],theta[3], theta[4],  mob,start,end)
        hospitalized_arr_grad= (np.array(hospitalized_grad))
        hospitalized_arr=(np.array(hospitalized))
        grad[i]=(hospitalized_arr_grad-hospitalized_arr)/0.0001
    return grad


class Multi_SIRH_model(Multi_Dimensional_Model):
    s_0=1000000 -1
    i_0=1
    r_0=0
    h_0=0
    dt=0.001
    def choose_model(self, gamma_constants):
        self.gamma_constants=gamma_constants
    def train(self,train_dates,data):
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
        self.train_dates=[i for i in range(len(data[0]))]
        mob = interp1d(np.arange(len(data[2])),data[2],kind="linear",fill_value="extrapolate")
        gamma_constants=self.gamma_constants
        if gamma_constants:
            self.gamma_constants=True
            curve1 = lambda x, a, b, h :   sirh_for_optim_m(x, a, b, h, mob)
            p,cov= curve_fit(curve1,[i for i in range(len(data[0]))],data[0], p0=[0,0,0],  bounds=([0,0,0], [np.inf,np.inf, np.inf]))
            self.a=p[0]
            self.b=p[1]
            self.h=p[2]
            self.cov=cov
            self.gamma_i=const_gamma_i
            self.gamma_h=const_gamma_h
        else :
            self.gamma_constants=False
            curve2=lambda x, a, b, h, gamma_i, gamma_h :   sirh_for_optim_m(x, a, b, h, mob, gamma_i=gamma_i, gamma_h=gamma_h)
            p,cov= curve_fit(curve2,[i for i in range(len(data[0]))],data[0], p0=[1e-2,1e-2,1e-2,1e-2,1e-2],  bounds=([0,0,0,0,0], [10,10, 10, 10,10]))
            self.a=p[0]
            self.b=p[1]
            self.h=p[2]
            self.gamma_i=p[3]
            self.gamma_h=p[4]
            self.cov=cov


        self.trained=True

    def point_predict(self,reach):
        mob=interp1d(np.arange(len(self.data[2])),self.data[2],kind="linear",fill_value="extrapolate")
        start=len(self.train_dates)
        end=start+reach
        s_0=1000000 -1
        i_0=1
        r_0=0
        h_0=0
        self.N=s_0 + i_0 + r_0 + h_0
        S,I,R,H=run_sirh_m([s_0, i_0, r_0, h_0], self.a, self.b,self.gamma_i, self.gamma_h,self.h ,mob,0,start)

        s_t=S[-1]
        i_t=I[-1]
        h_t=H[-1]
        r_t=R[-1]
        x0r=np.array([s_t, i_t, r_t, h_t])
        offset=H[-1]-self.data[0][-1]
        S,I,R,H=run_sirh_m(x0r, self.a, self.b,self.gamma_i, self.gamma_h,self.h ,mob,start,end)
        self.prediction=H-offset
        prediction=self.prediction

        return prediction

    def predict(self, reach,  alpha, method='covariance'):

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


        mob=interp1d(np.arange(len(self.data[2])),self.data[2],kind="linear",fill_value="extrapolate")
        start=len(self.train_dates)
        end=start+reach
        s_0=1000000 -1
        i_0=1
        r_0=0
        h_0=0
        self.N=s_0 + i_0 + r_0 + h_0
        S,I,R,H=run_sirh_m([s_0, i_0, r_0, h_0], self.a, self.b,self.gamma_i, self.gamma_h,self.h ,mob,0,start)

        #s0r=S[-1]
        #i0r=I[-1]
        #h0r=self.data[0][-1]
        #r0r=self.N-(s0r+i0r+h0r)
        #x0r=np.array([s0r,i0r,r0r,h0r])
        #assert self.trained, 'The model has not been trained yet'
        #S,I,R,H=run_sirh_m(x0r, self.a, self.b,self.gamma_i, self.gamma_h, self.h ,mob, start,end)
        #self.prediction=H

        s_t=S[-1]
        i_t=I[-1]
        h_t=H[-1]
        r_t=R[-1]
        x0r=np.array([s_t, i_t, r_t, h_t])
        offset=H[-1]-self.data[0][-1]
        S,I,R,H=run_sirh_m(x0r, self.a, self.b,self.gamma_i, self.gamma_h,self.h ,mob,start,end)
        self.prediction=H-offset
        prediction=self.prediction
        delta_method=True
        ci_low=0
        ci_high=0
        if delta_method:
            ci_low=[]
            ci_high=[]
            if self.gamma_constants:
                grad=grad_theta_h_theta_m(x0r, [self.a, self.b , self.h], mob,start,end) # size 3 x reach
            else :
                grad=grad_theta_h_theta_m_bis(x0r, [self.a, self.b ,  self.gamma_i, self.gamma_h, self.h], mob,start,end) # size 3 x reach

            cov=self.cov
            vars=np.diagonal((grad.transpose() @ cov @ grad).transpose())

            assert(len(vars)==reach), str(len(vars) + 'different from ' + str(reach))
            for i in range(len(vars)):
                down = scipy.stats.norm.ppf(alpha/2, loc=self.prediction[i], scale=np.sqrt(vars[i]))
                ci_low.append(down)
                up = scipy.stats.norm.ppf(1-(alpha/2), loc=self.prediction[i], scale=np.sqrt(vars[i]))
                ci_high.append(up)
            self.ci_low=ci_low
            self.ci_high=ci_high
        return prediction, [ci_low, ci_high]

def run_seir_m(x0,a,b,ph,mob,lag,t0,t1):
    T=np.arange(t0,t1)
    N=1000000
    solution=odeint(seir_m_rhs,x0,T,args=(a,b,N,mob,lag))
    S, E, I, R = solution.T  # Transpose and unpack
    return S, E, ph*I, R

def seir_m_rhs(x,t,a,b,N,mob,lag):
    x1, x2, x3, x4= x[0], x[1], x[2], x[3]
    dx1 =  -x1*(a*mob(t-lag)+b)*x3/N
    dx2 = x1*(a*mob(t-lag)+b)*x3/N - rho*x2
    dx3 = rho*x2 - gamma*x3
    dx4 = gamma*x3
    return [dx1, dx2, dx3, dx4]

def seir_for_optim(x,a,b,ph,mob):
    x0=[s_0, e_0, i_0, r_0]
    t=len(x)
    S,E,I,R=run_seir_m(x0,a,b,ph,mob,lag,0,t)
    return I

def grad_theta_seir(x0,theta,mob,start,end): # for gamma constant
    """
    compute the gradient of the estimation with respect to theta

    Parameters
    ----------
    x0 : np.array
        The initial state of the model
    theta : np.array
        The parameters of the model
    mob : np.array
        The mobility data estimated that will be used in the prediction
    start : int
        The start date
    end : int
        End date
    """
    reach=end-start
    grad=np.zeros((len(theta), reach))
    for i in range(len(grad)):
        theta_plus=theta.copy()
        theta_plus[i]+=0.0001
        _, _, I,_ = run_seir_m([x0[0], x0[1], x0[2], x0[3]], theta_plus[0], theta_plus[1], theta_plus[2],mob,lag,start,end)
        _, _, I_grad,_= run_seir_m([x0[0], x0[1], x0[2], x0[3]], theta[0], theta[1],theta[2],mob,lag,start,end)
        I_arr_grad= (np.array(I_grad))
        I_arr=(np.array(I))
        grad[i]=(I_arr_grad-I_arr)/0.0001
    return grad

class SEIR_model(Multi_Dimensional_Model):
    s_0=1000000 -1
    e_0=0
    i_0=1
    r_0=0
    def train(self,train_dates,data):
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
        self.train_dates=[i for i in range(len(data[0]))]
        mob = interp1d(np.arange(len(data[2])),data[2],kind="linear",fill_value="extrapolate")
        curve1 = lambda x, a, b, p :   seir_for_optim(x, a, b, p, mob)
        p,cov= curve_fit(curve1,[i for i in range(len(data[0]))],data[0], p0=[1,1,1],  bounds=([0,0,0], [np.inf,np.inf, 1]))
        self.a=p[0]
        self.b=p[1]
        self.ph=p[2]
        self.cov=cov
        self.trained=True

    def point_predict(self, reach):
        mob=interp1d(np.arange(len(self.data[2])),self.data[2],kind="linear",fill_value="extrapolate")
        start=len(self.train_dates)
        end=start+reach
        s_0=1000000 -1
        e_0=0
        i_0=1
        r_0=0
        self.N=s_0 + e_0 + i_0 + r_0
        S,E,I,R=run_seir_m([s_0, e_0, i_0, r_0], self.a, self.b,self.ph,mob,lag,0,start)
        s0r=S[-1]
        e0r=E[-1]
        i0r=I[-1]/self.ph #self.data[0][-1]/self.ph
        r0r=R[-1]#self.N-(s0r+e0r+i0r)
        offset=I[-1]-self.data[0][-1]
        x0r=np.array([s0r,e0r,i0r,r0r])
        assert self.trained, 'The model has not been trained yet'
        S,E,I,R=run_seir_m(x0r, self.a, self.b,self.ph,mob,lag,start,end)
        self.prediction=I-offset
        prediction=self.prediction

        return prediction

    def predict(self, reach,  alpha, method='covariance'):

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


        mob=interp1d(np.arange(len(self.data[2])),self.data[2],kind="linear",fill_value="extrapolate")
        start=len(self.train_dates)
        end=start+reach
        s_0=1000000 -1
        e_0=0
        i_0=1
        r_0=0
        self.N=s_0 + e_0 + i_0 + r_0
        S,E,I,R=run_seir_m([s_0, e_0, i_0, r_0], self.a, self.b,self.ph,mob,lag,0,start)
        s0r=S[-1]
        e0r=E[-1]
        i0r=I[-1]/self.ph #self.data[0][-1]/self.ph
        r0r=R[-1]#self.N-(s0r+e0r+i0r)
        offset=I[-1]-self.data[0][-1]
        x0r=np.array([s0r,e0r,i0r,r0r])
        assert self.trained, 'The model has not been trained yet'
        S,E,I,R=run_seir_m(x0r, self.a, self.b,self.ph,mob,lag,start,end)
        self.prediction=I-offset
        prediction=self.prediction
        delta_method=True
        ci_low=0
        ci_high=0
        if delta_method:
            ci_low=[]
            ci_high=[]
            grad=grad_theta_seir(x0r, [self.a, self.b , self.ph], mob,start,end) # size 3 x reach
            cov=self.cov
            vars=np.diagonal((grad.transpose() @ cov @ grad).transpose())

            assert(len(vars)==reach), str(len(vars) + 'different from ' + str(reach))
            for i in range(len(vars)):
                down = scipy.stats.norm.ppf(alpha/2, loc=self.prediction[i], scale=np.sqrt(vars[i]))
                ci_low.append(down)
                up = scipy.stats.norm.ppf(1-(alpha/2), loc=self.prediction[i], scale=np.sqrt(vars[i]))
                ci_high.append(up)
            self.ci_low=ci_low
            self.ci_high=ci_high
        return prediction, [ci_low, ci_high]
