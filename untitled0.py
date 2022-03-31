# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 12:30:07 2022

@author: jgull
"""

Y = [43, 57, 64, 69, 87, 87, 88, 91, 95, 106, 131]
import getters as get
X = x

#Y = get.get_cost_team(playerspldata, t)
mean_x = np.mean(X)
mean_y = np.mean(Y)



numm = []
denn = []
for i in range(len(X)):
  numm.append((X[i] - mean_x) * (Y[i] - mean_y))
  denn.append(X[i] - mean_x) ** 2)
ma = sum(numm) / sum(denn)
ca = mean_y - (ma * mean_x)

x = range(len(Y))
yt = ca + ma * x
print(ma,ca,numm,denn)

ss_t = 0 #total sum of squares
ss_r = 0 #total sum of square of residuals

for i in range(len(Y)): # val_count represents the no.of input x values
  ss_t += (Y[i] - mean_y) ** 2
  ss_r += (Y[i] - yt[i]) ** 2
print(ss_r)
print(ss_t)
r2 = 1 - (ss_r/ss_t)
print(r2)

#7.036363636363636 48.27272727272727 774.0 110.0
#482.58181818181833
#5928.727272727273
#0.9186027968596663

#%%
x=X
y=Y
mean_x = np.mean(x)
mean_y = np.mean(y)

sub_mean_x = np.subtract(x, mean_x)
sub_mean_y = np.subtract(y,mean_y)
print(sub_mean_x)
print(sub_mean_y)
prod = np.multiply(sub_mean_x,sub_mean_y)
print(len(prod), "l")
numer = np.sum(prod)
print(numer)
denom = np.sum(np.power(sub_mean_x,2))
print(denom)
m = np.divide(numer,denom)
c = mean_y - np.multiply(m,mean_x)
print(m,c, numer, denom)

y_pr = np.add(c,np.multiply(m,x))
ss_t = np.sum(np.power((y - mean_y),2))
ss_r = np.sum(np.power((y - y_pr), 2))  