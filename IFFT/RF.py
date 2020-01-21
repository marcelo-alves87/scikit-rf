import skrf as rf
from pylab import *
import matplotlib.pyplot as plt
import numpy as np
import csv

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


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

def get_return_loss_from_linear(data):
    return -20*np.log10(abs(data))

def get_return_loss_from_log(data):
    return abs(data)

def get_vswr_from_log(data):
    data = get_return_loss_from_log(data)
    data1 = 10**(-data/20)
    return (1 + data1) / (1 - data1) 

points = np.arange(0,101)
##x,y = normalize_csv('S11-LIN.csv')
##plt.plot(x,y,label='S11-LIN VNA')
##
##x,y = normalize_csv('S11-LOG.csv')
##plt.plot(x,y,label='S11-LOG VNA')
##
##x,y = normalize_csv('VSWR.csv')
##plt.plot(x,y,label='VNA')
####
##x,y = normalize_csv('S11-LOG.csv')
##plt.plot(x,get_vswr_from_log(y),label='VSWR')

##points = np.arange(0,101)
##probe = rf.Network('RL.s1p')
##x,y = probe.plot_s_db_time()
##x,y=x[100:201],y[100:201]
##plt.plot(x*3*10**8/2,abs(y),label='Analytic')
plt.rc('legend', fontsize=20)    # legend fontsize



x,y = normalize_csv('VSWR__SMO.csv')
plt.plot(0.9*x,y,label='Medicao+Smooth', linewidth=2, color='red')
##plt.xticks(np.arange(min(x), max(x), 0.1))
##
##x,y = normalize_csv('S11-VSWR.csv')
##plt.plot(x,y,label='S11 VSWR')

probe = rf.Network('S11_LOG_.s1p')
x,y = probe.plot_s_db_time(window=('kaiser', 6))
x,y=x[100:201],y[100:201]
plt.plot(0.9*x*3*10**8/2,smooth(abs(y),3),label='Simulacao',  linewidth=2)

##x,y = normalize_csv('VSWR__.csv')
##plt.plot(0.9*x,y,label='Medicao', linewidth=2)

##probe = rf.Network('RL.s1p')
##x,y = probe.plot_s_db_time(window=('boxcar'))
##x,y=x[100:201],y[100:201]
##plt.plot(x*3*10**8/2,abs(y),label='Boxcar')

##probe = rf.Network('RL.s1p')
##x,y = probe.plot_s_db_time(window=('triang'))
##x,y=x[100:201],y[100:201]
##plt.plot(x*3*10**8/2,abs(y),label='Triang')

##probe = rf.Network('RL.s1p')
##x,y = probe.plot_s_db_time(window=('blackman'))
##x,y=x[100:201],y[100:201]
##plt.plot(x*3*10**8/2,abs(y),label='Blackman')
##
##probe = rf.Network('RL.s1p')
##x,y = probe.plot_s_db_time(window=('hamming'))
##x,y=x[100:201],y[100:201]
##plt.plot(x*3*10**8/2,abs(y),label='Hamming')
##
##probe = rf.Network('RL.s1p')
##x,y = probe.plot_s_db_time(window=('hann'))
##x,y=x[100:201],y[100:201]
##plt.plot(x*3*10**8/2,abs(y),label='Hann')
##
##probe = rf.Network('RL.s1p')
##x,y = probe.plot_s_db_time(window=('bartlett'))
##x,y=x[100:201],y[100:201]
##plt.plot(x*3*10**8/2,abs(y),label='Bartlett')
##
##probe = rf.Network('RL.s1p')
##x,y = probe.plot_s_db_time(window=('flattop'))
##x,y=x[100:201],y[100:201]
##plt.plot(x*3*10**8/2,abs(y),label='Flattop')
##
##probe = rf.Network('RL.s1p')
##x,y = probe.plot_s_db_time(window=('parzen'))
##x,y=x[100:201],y[100:201]
##plt.plot(x*3*10**8/2,abs(y),label='Parzen')
##
##probe = rf.Network('RL.s1p')
##x,y = probe.plot_s_db_time(window=('bohman'))
##x,y=x[100:201],y[100:201]
##plt.plot(x*3*10**8/2,abs(y),label='Bohman')
####
##probe = rf.Network('RL.s1p')
##x,y = probe.plot_s_db_time(window=('blackmanharris'))
##x,y=x[100:201],y[100:201]
##plt.plot(x*3*10**8/2,abs(y),label='Blackmanharris')
##
##probe = rf.Network('RL.s1p')
##x,y = probe.plot_s_db_time(window=('nuttall'))
##x,y=x[100:201],y[100:201]
##plt.plot(x*3*10**8/2,abs(y),label='Nuttall')
##
##probe = rf.Network('RL.s1p')
##x,y = probe.plot_s_db_time(window=('barthann'))
##x,y=x[100:201],y[100:201]
##plt.plot(x*3*10**8/2,abs(y),label='Barthann')
##
##probe = rf.Network('RL.s1p')
##x,y = probe.plot_s_db_time(window=('gaussian', 0.07))
##x,y=x[100:201],y[100:201]
##plt.plot(x*3*10**8/2,abs(y),label='Gaussian')
##
##probe = rf.Network('RL.s1p')
##x,y = probe.plot_s_db_time(window=('slepian', 0.03))
##x,y=x[100:201],y[100:201]
###plt.plot(x*3*10**8/2,abs(y),label='Slepian')
####
##probe = rf.Network('RL.s1p')
##x,y = probe.plot_s_db_time(window=('chebwin', 100))
##x,y=x[100:201],y[100:201]
###plt.plot(x*3*10**8/2,abs(y),label='Chebwin')

##x,y = normalize_csv('DTF-VSWR.csv')
##x,y=x[0:101],y[0:101]
##plt.plot(x,y,label='VNA')

#plt.gca().invert_yaxis()

plt.xlabel('Meter (m)', fontsize=20)
plt.ylabel('VSWR', fontsize=20)
plt.title('H1D50-20')
plt.grid()
plt.legend()
plt.show()
