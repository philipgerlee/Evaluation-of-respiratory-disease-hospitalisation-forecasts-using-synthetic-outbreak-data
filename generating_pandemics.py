import numpy as np
import covasim as cv
import pandas as pd
from Mcmc import all_combinaison


mobility='''0
-0,0436877025562212
-0,142183890633573
-0,414264196497882
-0,490226517606742
-0,520691444229291
-0,601104998394856
-0,565468289968282
-0,500887028474234
-0,533539503417897
-0,489363271950639
-0,479732407975015
-0,525650201055078
-0,4254
-0,4325
-0,394
-0,53
-0,47
-0,5
-0,53
-0,56
-0,55
-0,53
-0,5
-0,47
-0,39
-0,35
-0,34
-0,34
-0,33
-0,33
-0,32
-0,31
-0,32
-0,31
-0,42
-0,41
-0,43
-0,45
-0,44
-0,44
-0,51
-0,51
-0,66
-0,77
'''.replace(',','.').split('\n')
start_day = '2020-03-02'
end_day   = '2021-01-01'
date_range = pd.date_range(start=start_day, end=end_day, freq='D')
all_days = cv.date_range(start_day, end_day)
floatmobility = [float(i) for i in mobility if i != '']
coef_mobility=[1+floatmobility[i] for i in range(len(floatmobility))]


coef_mobility_by_week=np.array([coef_mobility[i//7] for i in range(len(all_days))])
mob1=[1 for i in range(len(coef_mobility_by_week))]
mob2=[1+0.5*np.sin(2*np.pi*i/len(coef_mobility_by_week)) for i in range(len(coef_mobility_by_week))]
mob3=[1 for i in range(50)] + [ 0.4 for i in range(70)] + [ 1.2 for i in range(50)] + [1 for i in range(50)] + [ 0.4 for i in range(40)] + [ 1.2 for i in range(46)]
mobilities = [coef_mobility_by_week, mob1, mob2, mob3]


assert len(mob1)==len(mob2)==len(mob3)==len(coef_mobility_by_week)==306

def create_params_bis(combinaison): 
    coefs = [ 1 for _ in range(14)]
    for p in combinaison[0]: 
        coefs[p]=2
    for p in combinaison[1]: 
        coefs[p]=0.5

    params_custom = dict(
    pop_size=1000000,
    start_day='2020-03-01',
    end_day   = '2021-01-01', 
    pop_type='hybrid',
    beta=0.015,
    location='Sweden',
    pop_infected=10,
    dur={
        'exp2inf': {'dist':'lognormal_int', 'par1':4.5*coefs[0], 'par2':1.5}, # par 1 = mean of the log normal distrib, par 2 = std of the log normal distrib. par1 represents the expected value of the number of days between exposure and infection
        'inf2sym': {'dist':'lognormal_int', 'par1':1.1*coefs[1], 'par2':0.9},
        'sym2sev': {'dist':'lognormal_int', 'par1':6.6*coefs[2], 'par2':4.9},
        'sev2crit': {'dist':'lognormal_int', 'par1':1.5*coefs[3], 'par2':2.0},
        'asym2rec': {'dist':'lognormal_int', 'par1':8.0*coefs[4], 'par2':2.0},
        'mild2rec': {'dist':'lognormal_int', 'par1':8.0*coefs[5], 'par2':2.0},
        'sev2rec': {'dist':'lognormal_int', 'par1':18.1*coefs[6], 'par2':6.3},
        'crit2rec': {'dist':'lognormal_int', 'par1':18.1*coefs[7], 'par2':6.3},
        'crit2die': {'dist':'lognormal_int', 'par1':10.7*coefs[8], 'par2':4.8},
    }, 
    rel_symp_prob= 1.0*coefs[9],
    rel_severe_prob=1.0*coefs[10],
    rel_crit_prob=1.0*coefs[11],
    rel_death_prob=1.0*coefs[12]
    )
    return params_custom



def create_pandemic(dico, interventions):
    interventions_sim=cv.change_beta(days=all_days, changes=interventions, do_plot=False)
    mysimul = cv.Sim(dico, interventions=interventions_sim)
    mysimul.run()
    return np.array(mysimul.results['n_severe'])

for i in range(len(mobilities))  : 
    interventions = mobilities[i]
    combinaisons=all_combinaison([2, 4, 9, 10])
    for j in range(len(combinaisons)):
            print(i, j)
            print(combinaisons[j])
            combinaison = combinaisons[j]
            params=create_params_bis(combinaison)
            interventions_sim=cv.change_beta(days=all_days, changes=interventions, do_plot=False)
            mysim = cv.Sim(params, interventions=interventions_sim)
            mysim.run()
            df=pd.DataFrame([np.array(mysim.results['n_severe']), np.array(mysim.results['n_infectious']), np.array(interventions), np.array(mysim.compute_r_eff())])
            df.index=['n_hospitalized', 'n_infectious', 'mobility', 'R_eff']
            df.to_csv('./all_pandemics/pandemic_'+str(i)+'_'+str(j)+'.csv')