from Model import Model, Multi_Dimensional_Model
from scipy.optimize import curve_fit, minimize
import numpy as np
from scipy.optimize import differential_evolution
from useful_functions import differenciate
import pandas as pd
import sys
sys.path.append('./../')
import scipy.stats
from useful_functions import shift


# values for all the predictions : 

s_0=1000000 -1
i_0=1
r_0=0
d_0=0
dt=0.001


def derive(x, beta, N, gamma, d): # returns the derivative used to compute the SIR model
    s=x[0]
    i=x[1]
    r=x[2]
    deads=x[3]
    return np.array([-beta*s*i/N, beta*s*i/N - gamma*i - d * i , gamma*i , d * i ])



def run_sir(x0, beta, gamma,d,  t, dt):
    
    x=x0
    S=[x[0]]
    I=[x[1]]
    R=[x[2]]
    D=[x[3]] # deads 
    n_iter=int(t/dt)
    N=sum(x0)
    for i in range(n_iter):
        x=x+dt*derive(x, beta, N, gamma, d)
        S.append(x[0])
        I.append(x[1])
        R.append(x[2])
        D.append(x[3])
    s_final=[]
    i_final=[]
    r_final=[]
    d_final=[]
    time=np.linspace(0, t, int(t/dt) )
    for i in range(len(time)-1):
        if (time[i]-int(time[i]))<dt: 
            s_final.append(S[i])
            i_final.append(I[i])
            r_final.append(R[i])
            d_final.append(D[i])
    return s_final, i_final, r_final, d_final
    


def sir_for_optim(x, beta, gamma, d):
    # x is a list of dates (0 - 122)
    x0=[s_0, i_0, r_0, d_0]
    t=len(x)
    S,I,R,D=run_sir(x0, beta, gamma,d,  t, dt)
    zer=np.array([0])
    d_arr=np.array(D)
    return differenciate(np.concatenate((zer,d_arr))) # returns a value per day





def sir_for_optim_2(x, beta, d):
    # x is a list of dates (0 - 122)
    x0=[s_0, i_0, r_0, d_0]
    t=len(x)
    S,I,R,D=run_sir(x0, beta, 0.2,d,  t, dt)
    zer=np.array([0])
    d_arr=np.array(D)
    return differenciate(np.concatenate((zer,d_arr))) # returns a value per day


def grad_theta_h_theta(x0, theta, reach ): 
    grad=np.zeros((len(theta), reach))
    for i in range(len(grad)): 
        if len(theta)==2: 
            theta_plus=theta.copy()
            theta_plus[i]+=0.0001
            _, _, _, deads_grad = run_sir([x0[0], x0[1], x0[2], x0[3]], theta_plus[0], 0.2, theta_plus[1], reach+1, 0.001)
            _, _, _, deads = run_sir([x0[0], x0[1], x0[2], x0[3]], theta[0], 0.2, theta[1], reach+1, 0.001)
        elif len(theta)==3: 
            theta_plus=theta.copy()
            theta_plus[i]+=0.0001
            _, _, _, deads_grad = run_sir([x0[0], x0[1], x0[2], x0[3]], theta_plus[0],  theta_plus[1],theta_plus[2],  reach+1, 0.001)
            _, _, _, deads = run_sir([x0[0], x0[1], x0[2], x0[3]], theta[0], theta[1], theta[2],  reach+1, 0.001)
        d_arr_grad=np.array(differenciate(np.array(deads_grad)))
        d_arr=np.array(differenciate(np.array(deads)))
        grad[i]=(d_arr_grad-d_arr)/0.0001
    return grad



class SIRD_model_2(Model): 
    s_0=1000000 -1
    i_0=1
    r_0=0
    d_0=0
    dt=0.001
    def choose_model(self, gamma_constant): # has to be called before train 
        self.gamma_constant=gamma_constant
    def train(self, train_dates, data):
        self.name='SIRD'
        self.data=data
        self.train_dates=train_dates
        gamma_constant=self.gamma_constant
        if gamma_constant: 
            p,cov= curve_fit(sir_for_optim_2, self.train_dates,data, p0=[ 5.477e-01  , 5.523e-04],  bounds=([0,0], [np.inf,np.inf]))
            self.beta=p[0]
            self.d=p[1]
            self.gamma=0.2
            self.cov=cov
            self.trained= True
        else : 
            p,cov= curve_fit(sir_for_optim, self.train_dates,data, p0=[ 5.477e-01 , 2.555e-02 , 5.523e-04],  bounds=([0,0,0], [10,5,5]))
            self.beta=p[0]
            self.gamma=p[1]
            self.d=p[2]
            self.cov=cov
            self.trained= True

    def predict(self, reach, alpha):
        S,I,R,D=run_sir([s_0, i_0, r_0, d_0], self.beta, self.gamma, self.d , len(self.train_dates), 0.001)
        self.S=S
        self.I=I
        self.R=R
        self.D=D
        assert self.trained, 'The model has not been trained yet'
        deads=sir_for_optim(np.array([i for i in range(len(self.train_dates)+reach)]), self.beta, self.gamma,self.d)
        self.prediction =  deads[-reach:]
        prediction=self.prediction
        ci_low=[]
        ci_high=[]
        # We use the delta method to compute the confidence intervals
        if self.gamma_constant: 
            grad=grad_theta_h_theta([self.S[-1], self.I[-1], self.R[-1], self.D[-1]], [self.beta, self.d], reach) # size 2 x reach
        else : 
            grad=grad_theta_h_theta([self.S[-1], self.I[-1], self.R[-1], self.D[-1]], [self.beta,self.gamma,  self.d], reach) # size 3 x reach
        cov=self.cov
        vars=np.diagonal((grad.transpose() @ cov @ grad).transpose())
        assert(len(vars)==reach, str(len(vars)) + 'different from ' + str(reach))
        for i in range(len(vars)): 
            down = scipy.stats.norm.ppf(alpha/2, loc=self.prediction[i], scale=np.sqrt(vars[i]))
            ci_low.append(down)
            up = scipy.stats.norm.ppf(1-(alpha/2), loc=self.prediction[i], scale=np.sqrt(vars[i]))
            ci_high.append(up)
        self.ci_low=ci_low
        self.ci_high=ci_high
    
        return prediction, [ci_low, ci_high]
        




def run_sir_m(x0, a, b , gamma,d, mobility , dt):
    t=len(mobility)
    x=x0
    S=[x[0]]
    I=[x[1]]
    R=[x[2]]
    D=[x[3]] # deads 
    n_iter=int(t/dt)
    N=sum(x0)
    for i in range(n_iter):
        todays_mobility=mobility[int(i*dt)]
        beta=a*todays_mobility+b
        x=x+dt*derive(x, beta, N, gamma, d)
        S.append(x[0])
        I.append(x[1])
        R.append(x[2])
        D.append(x[3])
    s_final=[]
    i_final=[]
    r_final=[]
    d_final=[]
    time=np.linspace(0, t, int(t/dt) )
    for i in range(len(time)-1):
        if abs(time[i]-int(time[i]))<dt: 
            s_final.append(S[i])
            i_final.append(I[i])
            r_final.append(R[i])
            d_final.append(D[i])
    return s_final, i_final, r_final, d_final

def sir_for_optim_m( x, a, b ,d, mobility): # returns first the number of deaths and then the number of total infected
    
    s_0=1000000 -1
    i_0=1
    r_0=0
    d_0=0
    x0=np.array([s_0, i_0, r_0, d_0])
    dt=0.001

    S, I, R, D = run_sir_m(x0, a, b , 0.2,d, mobility ,   dt)
    zer=np.array([0])
    d_arr=np.array(D)
    I_arr=np.array(I)
    return np.concatenate((differenciate(np.concatenate((zer,d_arr))), I_arr))




def sir_for_optim_normalized(x, a, b, d, mobility, new_deaths, n_infected, shift1= 0, shift2 = 0 , taking_I_into_account=True): # returns firts the number of deaths and then the number of total infected
    I_and_D=sir_for_optim_m(x, a, b, d, mobility)
    I=I_and_D[len(I_and_D)//2:]
    D=I_and_D[:len(I_and_D)//2]
    if taking_I_into_account: 
        return np.concatenate((shift(D, shift1)/np.max(new_deaths), shift(I, shift2)/np.max(n_infected)))
    else:
        return D




def grad_theta_h_theta_m(x0, theta, mob_predicted ): 
    reach=len(mob_predicted) 
    grad=np.zeros((len(theta), reach))
    for i in range(len(grad)): 
        theta_plus=theta.copy()
        theta_plus[i]+=0.0001
        mob_extended=np.concatenate((mob_predicted, np.array([mob_predicted[-1]])))
        _, _, _, deads_grad = run_sir_m([x0[0], x0[1], x0[2], x0[3]], theta_plus[0], theta_plus[1], 0.2,theta_plus[2] , mob_extended, 0.001)
        _, _, _, deads= run_sir_m([x0[0], x0[1], x0[2], x0[3]], theta[0], theta[1], 0.2, theta[2], mob_extended, 0.001)
        d_arr_grad= np.array(differenciate(np.array(deads_grad)))
        d_arr=np.array(differenciate(np.array(deads)))
        grad[i]=(d_arr_grad-d_arr)/0.0001
    return grad

class Multi_SIRD_model(Multi_Dimensional_Model): 
    s_0=1000000 -1
    i_0=1
    r_0=0
    d_0=0
    dt=0.001
    def choose_model(self, taking_I_into_account, shifts, variation_of_shift1):
        if variation_of_shift1: 
            self.name='SIRD multi 1'
            self.range1=range(15)
            self.range2=range(1)
        else:
            self.name='SIRD multi 2'
            self.range1=range(1)
            self.range2=range(-15, 0)
            
        self.taking_I_into_account=taking_I_into_account
        self.shifts=shifts
        self.variation_of_shift1=variation_of_shift1

    def train(self, train_dates, data):
        self.data=data
        self.train_dates=train_dates
        if self.taking_I_into_account: 
            obj=np.concatenate((np.array(data[0])/max(np.array(data[0])), np.array(data[1])/max(np.array(data[1]))))
            coef=2
        else: 
            obj=np.array(data[0]/max(np.array(data[0])))
            coef=1
        if self.shifts: 
            method1=False
            if method1: 
                x=np.array([i for i in range(coef*len(train_dates))])
                curve2 = lambda params :     ((1- (params[3] - int(params[3]))) *np.sum(( sir_for_optim_normalized(x, params[0], params[1], params[2], self.data[1], self.data[0], self.data[1], shift1=int(params[3]), shift2= int(params[4]), taking_I_into_account=self.taking_I_into_account) - obj )**2)
                                            + ((params[3] - int(params[3]))) *np.sum(( sir_for_optim_normalized(x, params[0], params[1], params[2], self.data[1], self.data[0], self.data[1], shift1=int(params[3])+1, shift2= int(params[4]), taking_I_into_account=self.taking_I_into_account) - obj )**2)
                                            + (1-(params[4] - int(params[4]))) *np.sum(( sir_for_optim_normalized(x, params[0], params[1], params[2], self.data[1], self.data[0], self.data[1], shift1=int(params[3]), shift2= int(params[4]), taking_I_into_account=self.taking_I_into_account) - obj )**2)
                                            + ((params[4] - int(params[4]))) *np.sum(( sir_for_optim_normalized(x, params[0], params[1], params[2], self.data[2], self.data[0], self.data[1], shift1=int(params[3]), shift2= int(params[4])+1, taking_I_into_account=self.taking_I_into_account)- obj )**2))
                res=minimize(curve2, [1, 1, 5.523e-04, 5, 10],  bounds = [(-np.inf, np.inf), (-np.inf, np.inf), (0, np.inf), (-np.inf, np.inf), (-np.inf, np.inf)])
                p=res.x
                self.a=p[0]
                self.b=p[1]
                self.d=p[2]
                self.shift1=p[3]
                self.shift2=p[4]
                self.cov=res.hess_inv            
            else: 
                dico_losses=dict()
                best_result_so_far=np.inf
                best_p=None
                best_cov=None
                best_shift1=None
                best_shift2=None
                for shift1 in self.range1: 
                    for shift2 in self.range2:
                        curve1 = lambda x, a, b, d :   sir_for_optim_normalized(x, a, b, d, self.data[2], data[0], data[1], shift1=shift1, shift2= shift2, taking_I_into_account=self.taking_I_into_account)
                        try: 
                            p, cov= curve_fit(curve1,np.array([i for i in range(coef*len(train_dates))]),obj, p0=[ 1, 1 , 5.523e-04],  bounds=([-np.inf, -np.inf, 0], [np.inf,np.inf, np.inf]))
                            local_result=np.sum((curve1(np.array([i for i in range(coef*len(train_dates))]), p[0], p[1], p[2])-obj)**2)
                            dico_losses[str(shift1) + ' ' + str(shift2)]=local_result
                        except (RuntimeError, ValueError): 
                            local_result=np.inf
                            dico_losses[str(shift1) + ' ' + str(shift2)]=np.inf
                        if local_result<best_result_so_far:
                            best_result_so_far=local_result
                            best_p=p
                            best_cov=cov
                            best_shift1=shift1
                            best_shift2=shift2
                self.shift1=best_shift1
                self.shift2=best_shift2
                self.p=best_p
                self.a=self.p[0]
                self.b=self.p[1]
                self.d=self.p[2]
                self.cov=best_cov
                self.all_losses=dico_losses
        else:
            curve1 = lambda x, a, b, d :   sir_for_optim_normalized(x, a, b, d, self.data[2], data[0], data[1], shift1=0, shift2= 0, taking_I_into_account=self.taking_I_into_account)
            p,cov= curve_fit(curve1,np.array([i for i in range(coef*len(train_dates))]),obj, p0=[ 1, 1 , 5.523e-04],  bounds=([-np.inf, -np.inf, 0], [np.inf,np.inf, np.inf]))
            self.a=p[0]
            self.b=p[1]
            self.d=p[2]
            self.cov=cov
        self.gamma=0.2
        self.trained= True

    
    def predict(self, reach,  alpha):


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


        
        mob_predicted=np.array([self.data[2][-1] for i in range(reach)])
        reach=len(mob_predicted)
        s_0=1000000 -1
        i_0=1
        r_0=0
        d_0=0
        S,I,R,D=run_sir_m([s_0, i_0, r_0, d_0], self.a, self.b,0.2,  self.d ,self.data[2], 0.001)
        self.S=S
        self.I=I
        self.R=R
        self.D=D
        assert self.trained, 'The model has not been trained yet'
        deads_and_n_infected=sir_for_optim_m(None, self.a, self.b,self.d, np.concatenate((np.array(self.data[2]), mob_predicted)))
        deads=deads_and_n_infected[:len(np.array(self.data[2]))+len(mob_predicted)]
        if self.shift1==0: 
            self.prediction =  deads[-reach:] # shifting of shift1 for the prediction
        else : 
            self.prediction =  deads[-reach-self.shift1:-self.shift1] # shifting of shift1 for the prediction

        prediction=self.prediction
        
        ci_low=[]
        ci_high=[]
        grad=grad_theta_h_theta_m([self.S[-1], self.I[-1], self.R[-1], self.D[-1]], [self.a, self.b , self.d], mob_predicted) # size 3 x reach
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



