import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from numba import njit

temperature=310 #Kelvin
k_on=670 #1/seconds
k_off=130 #1/seconds
delta_t=0.000001 #seconds
D=40000 #nanometer^2/second

@njit         #Python compiler, useful to have a faster code
def force(x): #the first derivative of the potential
  xm=x%8      #period of 8 nm
  if(xm>=0 and xm<3):
    return -1.3
  elif(xm>=3 and xm<4):
    return 3.5
  elif(xm>=4 and xm<7):
    return -0.7
  else:
    return 2.5
  
@njit
def potential(x): #the potential
  xm=x%8          #period of 8 nm
  if(xm>=0 and xm<3):
    return -1.3*x
  if(xm>=3 and xm<4):
    return 3.5*x-14.5
  if(xm>=4 and xm<7):
    return -0.7*x+2.2
  else:
    return 2.5*x-20
  

@njit  
def mala_step_on(x, brownian):                          #integration step with check Metropolis-Hastings
    y=x-(force(x)*delta_t*D)+(math.sqrt(2*D)*brownian)  #candidate y
    
    log_pi_ratio=potential(x)-potential(y)
    mean_x=x-(force(x)*delta_t*D)
    mean_y=y-(force(y)*delta_t*D)
    denominator=1/(4*D*delta_t)
    log_q_ratio=denominator*((y-mean_x)*(y-mean_x)-(x-mean_y)*(x-mean_y))
    log_alpha=log_pi_ratio + log_q_ratio
    #for simplicity we calculate the logarithm of alpha

    if(math.log(np.random.uniform())<log_alpha):
      return y, False
    else:
      return x, True

@njit
def evolution(N, D): #evolves the system for N time steps
  t=0
  x=3.0 #start from a minimum of the potential
  state=0 # 0 means potential on, 1 means potential off

  for i in range(N):
    brownian=np.random.normal(0, math.sqrt(delta_t))

    if(state==0):
      x, not_accepted=mala_step_on(x, brownian)
    if(not_accepted): #don't update position and time if the transition is not accepted
      continue
    if(state==1):
      x=x+(math.sqrt(2*D)*brownian)
    t=t+delta_t

    if(state==0 and np.random.uniform()<=k_off*delta_t):  #changes of state
      state=1
      D=400
      continue
    if(state==1 and np.random.uniform()<=k_on*delta_t):
      state=0
      D=40000
      continue

  velocity=x/t
  return velocity

def gaussian(x, amp, mu, sigma):      #Gaussian function for fit
    return amp*np.exp(-(x-mu)*(x-mu)/(2*sigma*sigma))


# main

np.random.seed(54)   #set seed for reproducibility
velocities=np.zeros(100) # use 100 velocities to draw an histogram
for i in range(100):
  velocities[i]=evolution(10000000, D) #evolve for 10^7 time steps

n, bins, _ =plt.hist(velocities, bins=80, density=True) #histogram of velocities
centers = 0.5 * (bins[:-1] + bins[1:]) #take the central x for each bin

initial_guess=[n.max(), np.mean(velocities), np.std(velocities)]

amp, mu, sigma=curve_fit(gaussian, centers, n, p0=initial_guess)[0]

plt.plot(centers, gaussian(centers, amp, mu, sigma))
plt.title("Histogram of velocities with two-state system")
plt.xlabel("Velocity (nm/s)")
plt.ylabel("Count")
plt.show()

print("Mean velocity: " + str(mu))
print("Standard deviation: " + str(sigma))