from evolution_functions import evolution_position_time
import matplotlib.pyplot as plt

N=1000000
k_on=670 #1/seconds
k_off=130 #1/seconds
delta_t=0.000001 #seconds
D_on=40000 #nanometer^2/second
D_off=400 #nanometer^2/second


position, time=evolution_position_time(N, D_on, D_off, k_on, k_off, delta_t) #evolve for 10^7 time steps

plt.plot(time, position, color="green")
plt.title("Time evolution of the kinesin")
plt.xlabel("Time (s)")
plt.ylabel("Position (nm)")
plt.show()