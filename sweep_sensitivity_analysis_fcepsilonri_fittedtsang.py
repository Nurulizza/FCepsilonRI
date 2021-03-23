from collections import OrderedDict
import numpy as np
import math
from SALib.sample import saltelli
from SALib.analyze import sobol
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit, minimize,least_squares


import OpenCOR as oc

#50 samples
dict = {'FCepsilonRI/k_f1': 58.3902,
  'FCepsilonRI/k_f2': 0.00817889,
  'FCepsilonRI/k_f3': 1.0887,
  'FCepsilonRI/k_f4': 10.5797,
  'FCepsilonRI/k_f5': 63.6727,
  'FCepsilonRI/k_f6': 0.414288,
  'FCepsilonRI/k_f7': 11.4185,
  'FCepsilonRI/k_r1': 0.013548,
  'FCepsilonRI/k_r4': 0.0806961,
  'FCepsilonRI/k_r6': 0.731255,
  'FCepsilonRI/Pi': 1.89585}

  
varyme = 'FCepsilonRI/Pi'

# The state variable  or variables in the model that the data represents
num_series = 1
expt_state_uri = ['FCepsilonRI/pGrb2']

#Some example output that we are maybe aiming for
times = np.array([0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900,
                  2000, 2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800, 2900, 3000, 3100, 3200, 3300, 3400, 3500, 3600])
exp_data = np.zeros([num_series,len(times)])
exp_data[0,:] = np.array([0.0, 0.5, 0.97, 1.35, 1.8, 2.58, 2.95, 3.55, 3.82, 4.43, 4.7, 4.96, 5.15, 5.38, 5.6, 5.68, 5.87, 5.98,
                          6.02, 6.1, 6.11, 6.13, 6.24, 6.27, 6.12, 6.29, 6.44, 6.31, 6.34, 6.38, 6.37, 6.37, 6.36, 6.34, 6.35, 6.43, 6.47]) #pGrb2

#Number of samples to generate for each parameter
num_samples =  100

#Number of results to retain, if we store too many in high res parameter sweeps we can have memory issues
num_retain = 10

#List of parameters you want to exclude from fit
#fit_parameters_exclude = ['FCepsilonRI/pFC']

class Simulation(object):
    def __init__(self):
        self.simulation = oc.simulation()
        self.simulation.data().setStartingPoint(0)
        self.simulation.data().setEndingPoint(3600)
        self.simulation.data().setPointInterval(1)
        self.constants = self.simulation.data().constants()
        self.simulation.resetParameters()
        self.simulation.clearResults()
        for k,v in dict.items():
             self.constants[k]=v
        self.model_constants = OrderedDict({k: self.constants[k]
                                            for k in self.constants.keys()})  
		
    
    def run_once(self, c, v):
        self.simulation.resetParameters()
        self.constants[c] = v
        self.simulation.run()
        return (self.simulation.results().points().values(),
                self.simulation.results().states()['FCepsilonRI/pGrb2'].values())
    
    def run_sensitvity(self, c, scale=2.0):
        self.simulation.clearResults()
        v = self.model_constants[c]
        base = self.run_once(c, v)[1][times]
        divergence = 0.0
        for s in [1.0/scale, scale]:
            trial = self.run_once(c, s*v)[1][times]
            divergence += math.sqrt(np.sum((base - trial)**2))
        return divergence
    
    def evaluate_model(self, parameter_values):
        self.simulation.clearResults()
        for i, k in enumerate(self.model_constants.keys()):
            self.constants[k] = parameter_values[i]
        #print('Parameter set: ', parameter_values)
        self.simulation.run()
        return (self.simulation.results().states()['FCepsilonRI/pGrb2'].values()[times])
    
    def evaluate_ssq(self):
        self.simulation.clearResults()

        self.simulation.run()
        trial = np.zeros([num_series,len(times)])
        ssq = np.zeros(num_series+1)
		
        for i in range(0,num_series):
            trial[i,:] = self.simulation.results().states()[expt_state_uri[i]].values()[times]
            ssq[i+1] = math.sqrt(np.sum((exp_data[i,:]-trial[i,:])**2))
        ssq[0] = np.sum(ssq[1:num_series+1])
        return ssq 
        
    
    def run_parameter_sweep(self):
        num_cols = num_series + 1 + self.samples.shape[1]
        num_rows = num_retain+1
        Y = np.zeros([num_rows,num_cols])
        for i, X in enumerate(self.samples):
            ssq = self.evaluate_ssq(X)
            j = i
            if j < num_retain:
                Y[j,0] = ssq[0]
                for k in range(0,num_series):
                    Y[j,k+1] = ssq[k+1]
                Y[j,(k+2):num_cols]=X
            else:
                Y[num_retain,0] = ssq[0]
                for k in range(0,num_series):
                    Y[num_retain,k+1] = ssq[k+1]
                Y[num_retain,(k+2):num_cols]=X
                ind = np.argsort(Y[:,0])
                Y=Y[ind]
				
			#Want to retain top N here
        ind = np.argsort(Y[:,0])
        Z=Y[ind]
        return Z			


plt.close('all')

plt.semilogx([dict[varyme],dict[varyme]],[0,1],label='fit value')
values = np.zeros([3,len(np.arange(-3,2,0.1))])
count = 0
for i in np.arange (-3,2,0.1):
      dict[varyme] = 10**i
      
      s = Simulation()
      
      v = s.evaluate_ssq()
      values[0,count] = 10**i
      values[1:3,count] = v
      count = count +1
      print(count,values[0,count-1])
print(values)
plt.semilogx(values[0,:],values[1,:],label='pGrb2', color='red')
plt.legend()
plt.xlabel(varyme)
plt.ylabel('Error metric')

plt.show()
