import numpy as np
import csv
import matplotlib as mpl
#mpl.use('pdf')
import matplotlib.pyplot as plt

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

x,y=normalize_csv('PSO/PSO_1.csv')

plt.rc('font', family='arial')
plt.rc('xtick', labelsize='x-small')
plt.rc('ytick', labelsize='x-small')
plt.rc('text', usetex=True)
fig = plt.figure(figsize=(4, 4))
ax = fig.add_subplot(1, 1, 1)

#x = np.linspace(0.0, 5, 201)
ax.plot(x, y, color='k', ls='solid')
#ax.plot(x, 20/x, color='0.50', ls='dashed')
ax.set_xlabel('Metro (m)')
ax.set_ylabel('VSWR')
plt.grid()
plt.show()

