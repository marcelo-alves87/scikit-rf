import skrf as rf
from pylab import *
import matplotlib.pyplot as plt
import numpy as np
import csv

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


def normalize_csv(filename, param):
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
                    ys.append(row[param])
                else:
                    ys.append(row[param])
    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)

def get_return_loss_from_linear(data):
    return -20*np.log10(abs(data))

def get_return_loss_from_log(data):
    return abs(data)

def get_vswr_from_log(data):
    data = get_return_loss_from_log(data)
    data1 = 10**(-data/20)
    return (1 + data1) / (1 - data1) 


##x,y = normalize_csv('H1D75/H1D75_Simulacao.csv', 1)
##plt.plot(x*10**9,y, label = 'Simulacao')
##
##x,y = normalize_csv('H1D75/H1D75-Med-Freq.csv', 1)
##plt.plot(x,y,label='Medicao')

##x,y = normalize_csv('H1D50-20 - Equivalente - Modulo.csv', 1)
##x,z = normalize_csv('H1D50-20 - Equivalente - Fase.csv', 1)
##
##x = [int(x1*10**9) for x1 in x]
##
##f = open("H1D50-20-Equivalente.s1p","w+")
##f.write('!Keysight Technologies N9923A: A.08.19\n')
##f.write('!Date: Thursday, 16 January 2020 11:30:00\n')
##f.write('!TimeZone: (GMT-03:00) Brasilia!Model: N9923A\n')
##f.write('!Serial: MY51491501\n')
##f.write('!GPS Latitude: \n')
##f.write('!GPS Longitude: \n')
##f.write('!GPS TimeStamp: 0001-01-01 00:00:00Z\n')
##f.write('!GPS Seconds Since Last Read: 0\n')
##f.write('!CHECKSUM:1078980270\n')
##f.write('!Correction: S11(ON U)\n')
##f.write('!S1P File: Measurement: S11:\n')
##f.write('# Hz S DB R 50\n')
##
##for i in range(len(x)):
##    f.write(str(x[i]) + ' ' + str(y[i]) + ' ' + str(z[i]) + '\n')
## 
##f.close()


probe = rf.Network('H1D50-20-Equivalente.s1p')
x,y = probe.plot_s_db_time(window=('kaiser', 6))
plt.plot(3*10**8*x/2, y,label='Simulacao',  linewidth=2)
##
##x,y = normalize_csv('H1D75/H1D75-Med-DTF_4.CSV', 1)
##plt.plot(0.87*x,smooth(y,3),label='Medicao', linewidth=2)

#plt.xticks(np.arange(min(x), max(x), 0.1))


##probe = rf.Network('S11_LOG_.s1p')
##x,y = probe.plot_s_db_time(window=('kaiser', 6))
##y = smooth(y,2)
##plt.plot(3*10**8*x, y ,label='Medicao',  linewidth=2)

##x,y = normalize_csv('VSWR__.csv', 1)
##plt.plot(0.9*x,smooth(y,3),label='Medicao', linewidth=2)

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

##plt.gca().invert_yaxis()
plt.rc('legend', fontsize=20)    # legend fontsize
plt.xlabel('Meter (m)', fontsize=20)
plt.ylabel('VSWR', fontsize=20)
plt.title('H1D75')
plt.grid()
plt.legend()
plt.show()
