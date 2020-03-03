import skrf as rf
from pylab import *
import matplotlib.pyplot as plt
import numpy as np
import csv
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)


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
               if end_index == param: 
                   break
               else:
                   xs = []
                   ys = []               
            elif(s.find('!') < 0 and s.find('BEGIN') < 0 and s.find('END') < 0) and s.find('Freq') < 0:
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


##x,y = normalize_csv('H1D15/H1D15_Simulacao.csv', 1)
##plt.plot(x*10**9,y, label = 'Simulacao')
##
##x,y = normalize_csv('H1D15/H1D15-Med-Freq.csv', 1)
##plt.plot(x,y,label='Medicao')

##for i in range(1,15):
##    
##    x,y = normalize_csv('CampoExperimental/' + str(i) +'Z.csv', 1)
##    x,z = normalize_csv('CampoExperimental/'+ str(i) + 'Z.csv', 2)
##
##    ##x = [int(x1*10**9) for x1 in x]
##
##    f = open("CampoExperimental/"+ str(i) + "Z.s1p","w+")
##    f.write('!Keysight Technologies N9923A: A.08.19\n')
##    f.write('!Date: Thursday, 16 January 2020 11:30:00\n')
##    f.write('!TimeZone: (GMT-03:00) Brasilia!Model: N9923A\n')
##    f.write('!Serial: MY51491501\n')
##    f.write('!GPS Latitude: \n')
##    f.write('!GPS Longitude: \n')
##    f.write('!GPS TimeStamp: 0001-01-01 00:00:00Z\n')
##    f.write('!GPS Seconds Since Last Read: 0\n')
##    f.write('!CHECKSUM:1078980270\n')
##    f.write('!Correction: S11(ON U)\n')
##    f.write('!S1P File: Measurement: S11:\n')
##    f.write('# Hz S DB R 50\n')
##
##    for i in range(len(x)):
##        f.write(str(x[i]) + ' ' + str(y[i]) + ' ' + str(z[i]) + '\n')
##     
##    f.close()

##

##fig, ax = plt.subplots()
##
##
##for i in [14]:
##    probe = rf.Network('CampoExperimental/'+str(i)+'Z.s1p')
##    x1,y1 = probe.plot_s_db_time(window=('kaiser', 6))
##    x1 = 0.92*3*10**8*x1/2
##
##    ax.plot(x1, y1,label='H' + str(i),linewidth=2)
##
##probe = rf.Network('Pontos/H1D50-20-201.s1p')
##x1,y1 = probe.plot_s_db_time(window=('kaiser', 6))
##x1 = 0.87*3*10**8*x1/2
##ax.plot(x1, smooth(y1,2),label='201',linewidth=2)

##x2,y2 = normalize_csv('Viagem-2-STN/Viagem-2-STN/31-07-2019/05v6/10-2/D/T13.csv', 1)
##ax.plot(x2,smooth(y2,3),label='Tempo', linewidth=2)
##
####majorLocator = MultipleLocator(0.2)
####majorFormatter = FormatStrFormatter('%.2f')
####minorLocator = MultipleLocator(0.04)
####
####ax.xaxis.set_major_locator(majorLocator)
####ax.xaxis.set_major_formatter(majorFormatter)
####ax.xaxis.set_minor_locator(minorLocator)
##ax.set_xticks(np.arange(0,11,0.5))

##x2,y2 = normalize_csv('Viagem-2-STN/Viagem-2-STN/31-07-2019/05v6/10-2/D/T13.csv', 1)
##plt.plot(x2,smooth(y2,3),label='Tempo', linewidth=2)

#plt.xticks(np.arange(0, max(x1),0.1),fontsize=6)


probe = rf.Network('H1N - Equivalente/H1N-Equivalente.s1p')
x,y = probe.plot_s_db_time(window=('kaiser', 6))
plt.plot(3*10**8*x/2,smooth(y,3),label='Simulacao',  linewidth=2)

x,y = normalize_csv('MDSC2/04-02-2019/H1N/H1N-Med-DTF-3_Amp.csv', 1)
plt.plot(x,smooth(y,3),label='Medicao', linewidth=2)

plt.xticks(np.arange(0,2,0.1))
####
##
##x1,y1 = normalize_csv('MDSC2/04-02-2019/H1D15/H1D15-Med-DTF-2_Amp.csv', 1)
##
##
##x2,y2 = normalize_csv('MDSC2/05-02-2019/H1D15/H1D15-Med-DTF-2_Amp.csv', 1)
##
##
##x3,y3 = normalize_csv('MDSC2/10-02-2019/H1D15/H1D15-Med-DTF-2_Amp.csv', 1)
##
##x4,y4 = normalize_csv('MDSC2/11-02-2019/H1D15/H1D15-Med-DTF-2_Amp.csv', 1)
##
##x = np.mean([x1,x2,x3,x4], axis=0)
##y = np.mean([y1,y2,y3,y4], axis=0)
##
##plt.plot(0.87*x,smooth(y,3),label='Media de Medicoes', linewidth=2)
##
##plt.xticks(np.arange(min(x), max(x), 0.1))



##
plt.rc('legend', fontsize=20)    # legend fontsize
plt.xlabel('Metro (m)', fontsize=20)
plt.ylabel('VSWR', fontsize=20)
plt.title('H1N')
plt.grid()
plt.legend()
plt.show()
