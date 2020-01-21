import numpy as np
import matplotlib as mpl
import csv
#mpl.use('pdf')
import matplotlib.pyplot as plt
import skrf as rf
from pylab import *

def normalize_csv(filename):
    xs = []
    ys = []
    end_index = 0
    
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            s = row[0]
            if s.find('END') >= 0:
               end_index += 1 
               #1 S11, 2 Phase, 3 SWR, 4 Real/Imaginary 
               if end_index == 1: 
                   break
               else:
                   xs = []
                   ys = []               
            elif(s.find('!') < 0 and s.find('BEGIN') < 0 and s.find('END') < 0):
                xs.append(row[0])
                if len(row) > 1:
                    #1 Real, 2 Imaginary
                    ys.append(row[1])
                else:
                    ys.append(row[1])
    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)

probe = rf.Network('RL.s1p')
x,y = probe.plot_s_db_time(window=('kaiser', 20))

plt.rc('font', family='sans')
plt.rc('xtick', labelsize='x-small')
plt.rc('ytick', labelsize='x-small')
plt.rc('text', usetex=True)
fig = plt.figure(figsize=(5, 4))
ax = fig.add_subplot(1, 1, 1)

xs,ys = probe.plot_s_db_time(window=('kaiser', 6))

#x = np.linspace(1., 8., 30)
ax.plot(x*3*10**8*0.79/2, y, color='green', label='Kaiser(20)')
#ax.plot(x*3*10**8*0.79/2, ys, color='black', label='Kaiser(6)')
ax.set_xlabel('Meter (m)', fontsize=12)
ax.set_ylabel('VSWR', fontsize=12)
#plt.gca().invert_yaxis()
plt.legend()
plt.grid()
plt.show()
