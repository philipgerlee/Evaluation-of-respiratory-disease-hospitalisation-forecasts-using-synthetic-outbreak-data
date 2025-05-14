import numpy as np
import sys 
sys.path.append('./models/')
from useful_functions import differenciate
import covasim as cv
import json
import sys
import time 

t0=time.time()

def diff_between_2_arrays(array1, array2): 
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
    res.append(np.sum([abs(ar1_normalized[i]-ar2_normalized[i]) for i in range(len(ar1_normalized))]))
    res.append(np.sum([abs(der1_normalized[i]-der2_normalized[i]) for i in range(len(der1_normalized))]))
    res.append(np.sum([abs(derder1_normalized[i]-derder2_normalized[i]) for i in range(len(derder1_normalized))]))
    res=np.array(res)
    return np.sum(res**2)
            
def all_combinaison(params): 
    p1=params[0]
    p2=params[1]
    p3=params[2]
    p4=params[3]

    res = [[[], []] for _ in range(81)]
    for i in range(3): 
        for j in range(3): 
            for k in range(3): 
                for l in range(3): 
                    n=27*i + 9*j + 3 * k + l
                    if i == 0 : 
                        res[n][0].append(p1)
                    elif i == 1 : 
                        res[n][1].append(p1)
                    if j == 0 : 
                        res[n][0].append(p2)
                    
                    elif j == 1 :
                        res[n][1].append(p2)
                    if k == 0 :
                        res[n][0].append(p3)
                    elif k == 1 :
                        res[n][1].append(p3)
                    if l == 0 :
                        res[n][0].append(p4)
                    elif l == 1 :
                        res[n][1].append(p4)
    return res

def create_params(combinaison): 
    coefs = [ 1 for _ in range(14)]
    for p in combinaison[0]: 
        coefs[p]=2
    for p in combinaison[1]: 
        coefs[p]=0.5

    params_custom = dict(
    pop_size=10000,
    start_day='2020-03-01',
    end_day='2021-03-01',
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




def create_pandemic(dico):
    mysimul = cv.Sim(dico)
    mysimul.run()
    return np.array(mysimul.results['n_severe'])




def loss(params): # params are the numero of the parameters to change 
    combinaisons=all_combinaison(params)
    pandemics=[create_pandemic(create_params(combinaisons[i])) for i in range(len(combinaisons))]
    loss=0
    for i in range(len(combinaisons)): 
        for j in range(i+1, len(combinaisons)): 
            loss+=diff_between_2_arrays(pandemics[i], pandemics[j])
    return loss

path_suivi='./results/suivi_3.txt'
path_dicoloss='./results/dicoloss_mcmc_3.json'
path_dicocount='./results/dicocount_mcmc_3.json'





if __name__ == '__main__': 
    arg = sys.argv[1]
    if arg == 'begin': 
        params_init=[0, 5, 10, 12]
        dicocount=dict()
        dicoloss=dict()
    elif arg == 'continue':
        with open('./results/last_params.txt', 'r') as f : 
            params_init = eval(f.read())
        with open (path_dicocount) as f :
            dicocount=json.load(f)
        with open (path_dicoloss) as f :
            dicoloss=json.load(f)

    else :
        print('WHERE PARAMETER ')
        exit()
        



    params=params_init
    if str(params_init) not in dicoloss.keys(): 
        loss_init=loss(params_init)
        dicoloss[str(params_init)]=loss_init
        dicocount[str(params_init)]=0
    else : 
        loss_init=dicoloss[str(params_init)]
        dicocount[str(params_init)]+=1

    with open(path_suivi, 'a') as f : 
        f.write('Initial parameters : '+str(params_init)+'\n')
        f.write('Initial loss : '+str(loss_init)+'\n')
        f.write('   \n')


    for n in range(200): 

       
            
        dicocount[str(params)]+=1
        
        index=np.random.randint( 4)
        new_param=[i for i in range(14) if i not in params] [np.random.randint(10)]
        new_params=params.copy()
        new_params[index]=new_param
        new_params.sort() # important to avoid counting the same set in different keys
        with open(path_suivi, 'a') as f : 
            f.write('Step number : '+str(n)+'\n')
            f.write(' new param selected : '+str(new_param)+'\n')
            f.write('It will replace ' + str(params[index]) + '\n')
            f.write('the new set is : '+str(new_params)+'\n')
            if str(new_params)  not in dicoloss.keys(): 
                f.write(' we never met this set before \n')
            else :
                f.write(' we already met this set before \n')
        loss_previous=dicoloss[str(params)]
        if str(new_params)  in dicoloss.keys(): 
            loss_new=dicoloss[str(new_params)]
        else :
            loss_new=loss(new_params)
            dicoloss[str(new_params)]=loss_new
            dicocount[str(new_params)]=0
        with open(path_suivi, 'a') as f :
            f.write('Previous loss : '+str(loss_previous)+'\n')
            f.write('New loss : '+str(loss_new)+'\n')
            f.write('   \n')
        changed = False 
        if loss_new > loss_previous: # attention, we want to increase dissemblance !!
            params=new_params
            changed=True
            with open(path_suivi, 'a') as f :
                f.write('The new loss is bigger \n')

        else : 
            p=np.random.rand()
            with open(path_suivi, 'a') as f :
                f.write( 'the ratio is ' + str(loss_new/loss_previous)+'\n' )
                f.write('p is : '+str(p)+'\n')
            if p<loss_new/loss_previous: 
                params=new_params
                changed=True
            else : 
                with open(path_suivi, 'a') as f :
                    f.write('The new set is rejected  \n')
                    f.write('\n')
                    f.write('\n')
                    f.write('\n')
        

        if changed:
            with open(path_suivi, 'a') as f :
                f.write('The new set is accepted \n')
                f.write('The new set is : '+str(params)+'\n')
                f.write('   \n')
                f.write('\n')
                f.write('\n')
                f.write('\n')
        
        # save dicos : 
        with open(path_dicoloss, 'w') as f : 
            json.dump(dicoloss, f)
        with open(path_dicocount, 'w') as f :
            json.dump(dicocount, f)

    # saving last parameters : 
    with open('./results/last_params.txt', 'w') as f : 
        f.write(str(params))
