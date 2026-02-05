import math
import numpy as np
import matplotlib.pyplot as plt
from numba import njit

temperature=310 #Kelvin
L=8 #nanometers
k_on=0 #1/seconds
k_off=0 #1/seconds
delta_t=0.00001 #seconds
D=20000 #nanometer^2/second

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

#@njit  
#def force(x): #the first derivative of the potential
  #xm=x%8
  #if(xm>=0 and xm<=0.1):
   # return 0
  #elif(xm>0.1 and xm<2.9):
  #  return -1.43
  #elif(xm>=2.9 and xm<=3.1):
  #  return 0
  #elif(xm>3.1 and xm<3.9):
  #  return 4.38
  #elif(xm>=3.9 and xm<=4.1):
  #  return 0
 # elif(xm>4.1 and xm<6.9):
 #   return -0.71
 # elif(xm>=6.9 and xm<=7.1):
  #  return 0
 # elif(xm>7.1 and xm<7.9):
 #   return 3.13
 # elif(xm>=7.9 and xm<=8):
 #   return 0
 # else:
 #   return 0 #but this case shouldn't happen

@njit
def evolution(N, D): #evolves the system for N time steps
  t=0
  x=3.0
  #position=np.zeros(N)
  #time=np.zeros(N)
  state=0 # 0 means potential on, 1 means potential off
  for i in range(N):
    brownian=np.random.normal(0, math.sqrt(delta_t))
    if(state==0):
      x, not_accepted=mala_step_on(x, brownian)
    if(not_accepted):
      continue
    if(state==1):
      x=x+(math.sqrt(2*D)*brownian)
    t=t+delta_t
    #position[i]=x
    #time[i]=t
    if(state==0 and np.random.uniform()<=k_off*delta_t):
      state=1
      #D=200
      continue
    if(state==1 and np.random.uniform()<=k_on*delta_t):
      state=0
      #D=20000
      continue
  velocity=x/t
  return velocity


speed=np.zeros(100) # use 100 velocities to draw an histogram
for i in range(100):
  speed[i]=evolution(10000000, D)

#position, time=evolution(100000000)

#plt.plot(time,position) #position vs time for the last value of the speed vector
#plt.xlabel("Time (s)")
#plt.ylabel("Position (nm)")
#plt.show()

plt.hist(speed, bins=100) #histogram of speed vector
plt.xlabel("Velocity (nm/s)")
plt.show()

print(np.mean(speed))
print(np.std(speed))

#print(evolution(100000000, D))
