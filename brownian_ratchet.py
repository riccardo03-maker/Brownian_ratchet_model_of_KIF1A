import math
import numpy as np
import matplotlib.pyplot as plt
from numba import njit

temperature=310 #Kelvin
k_on=670 #1/seconds
k_off=130 #1/seconds
delta_t=0.000001 #seconds
D=40000 #nanometer^2/second

@njit         #Python compiler, useful to have a faster code
def force(x): #the first derivative of the potential
  xm=x%8
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
  xm=x%8
  if(xm>=0 and xm<3):
    return -1.3*x
  if(xm>=3 and xm<4):
    return 3.5*x-14.5
  if(xm>=4 and xm<7):
    return -0.7*x+2.2
  else:
    return 2.5*x-20
  

@njit  
def mala_step_on(x, brownian):
    y=x-(force(x)*delta_t*D)+(math.sqrt(2*D)*brownian)
    log_pi_ratio=potential(x)-potential(y)

    mean_x=x-(force(x)*delta_t*D)
    mean_y=y-(force(y)*delta_t*D)
    denominator=1/(4*D*delta_t)

    log_q_ratio=denominator*((y-mean_x)*(y-mean_x)-(x-mean_y)*(x-mean_y))

    log_alpha=log_pi_ratio + log_q_ratio
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
    if(state==0 and np.random.uniform()<=k_off*delta_t):
      state=1
      D=400
      continue
    if(state==1 and np.random.uniform()<=k_on*delta_t):
      state=0
      D=40000
      continue
  velocity=x/t
  return velocity


np.random.seed(54)
speed=np.zeros(100) # use 100 velocities to draw an histogram
for i in range(100):
  speed[i]=evolution(10000000, D)

plt.hist(speed, bins=100) #histogram of speed vector
plt.xlabel("Velocity (nm/s)")
plt.show()

print(np.mean(speed))
print(np.std(speed))
