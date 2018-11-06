import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as dates

                         

vert_profile_levels_top   = [20, 92, 184, 324, 522, 781, 1106]
vert_profile_mid = [10]+[(vert_profile_levels_top[i]+vert_profile_levels_top[i+1])/2. for i in range(6)]
vert_profile_industry = [0.06, 0.16, 0.75, 0.03, 0, 0, 0]

hod = [0.75, 0.75, 0.78, 0.82, 0.88, 0.95, 1.02, 1.09, 1.16, 1.22, 1.28, 1.3, 1.22, 1.24, 1.25, 1.16, 1.08, 1.01, 0.95, 0.9, 0.85, 0.81, 0.78, 0.75]

dow = [1.08, 1.08, 1.08, 1.08, 1.08, 0.8, 0.8]

moy = [1.1, 1.075, 1.05, 1, 0.95, 0.9, 0.93, 0.95, 0.97, 1, 1.025, 1.05]

######### 
## Moy ##
#########
ax = plt.axes()
ax.plot(range(1,13),moy,"-o")
ax.set_xticks(range(1,13))
ax.set_xticklabels(["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"])
plt.savefig("moy.png")
plt.clf()

######### 
## Dow ##
#########
ax = plt.axes()
ax.plot(range(1,8),dow,"-o")
ax.set_xticks(range(1,8))
plt.savefig("dow.png")
plt.clf()

######### 
## Hod ##
#########
ax = plt.axes()
ax.plot(hod,"-o")
ax.set_xticks(range(0,24,2))
plt.savefig("hod.png")
plt.clf()

######### 
## Ver ##
#########
ax = plt.axes()
ax.plot(vert_profile_industry,vert_profile_mid,"-o")
ax.set_yticks(vert_profile_levels_top)
ax.grid(True,"major","y")
plt.savefig("ver.png")
