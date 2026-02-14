import math
import numpy as np
from numba import njit

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
def metropolis_step_on(x, brownian, delta_t, D):                          #integration step with check Metropolis-Hastings
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
def evolution_velocity(N, D_on, D_off, k_on, k_off, delta_t): #evolves the system for N time steps
  t=0
  x=3.0 #start from a minimum of the potential
  D=D_on
  state=0 # 0 means potential on, 1 means potential off

  for i in range(N):
    brownian=np.random.normal(0, math.sqrt(delta_t))

    if(state==0):
      x, not_accepted=metropolis_step_on(x, brownian, delta_t, D)
    if(not_accepted): #don't update position and time if the transition is not accepted
      continue
    if(state==1):
      x=x+(math.sqrt(2*D)*brownian)
    t=t+delta_t

    if(state==0 and np.random.uniform()<=k_off*delta_t):  #changes of state
      state=1
      D=D_off
      continue
    if(state==1 and np.random.uniform()<=k_on*delta_t):
      state=0
      D=D_on
      continue

  velocity=x/t
  return velocity

def gaussian(x, amp, mu, sigma):      #Gaussian function for fit
    return amp*np.exp(-(x-mu)*(x-mu)/(2*sigma*sigma))

def evolution_position_time(N, D_on, D_off, k_on, k_off, delta_t): #to get the plot position/time
    t=0
    x=3.0 #start from a minimum of the potential
    position=np.zeros(N) #vectors of position and time
    time=np.zeros(N)
    D=D_on
    state=0 # 0 means potential on, 1 means potential off
    index=0

    for i in range(N):
      brownian=np.random.normal(0, math.sqrt(delta_t))

      if(state==0):
        x, not_accepted=metropolis_step_on(x, brownian, delta_t, D)
      if(not_accepted): #don't update position and time if the transition is not accepted
        continue
      if(state==1):
        x=x+(math.sqrt(2*D)*brownian)

      t=t+delta_t
      position[index]=x
      time[index]=t
      index=index+1

      if(state==0 and np.random.uniform()<=k_off*delta_t):  #changes of state
        state=1
        D=D_off
        continue
      if(state==1 and np.random.uniform()<=k_on*delta_t):
        state=0
        D=D_on
        continue
    
    position=position[:-(N-index)]
    time=time[:-(N-index)]
    return position, time
    #removes all 0s at the end of the vectors. The number of 0s depends on the number of transitions rejected
