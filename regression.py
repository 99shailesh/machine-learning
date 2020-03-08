#------------------------------------------------------------------
import math as m
import numpy as np 
import pandas as pd 
from matplotlib import pyplot as plt
#------------------------------------------------------------------

#-------------------------------------------------------------------
# importing data and processing it
#-------------------------------------------------------------------
dataset=pd.read_csv("iris.csv")
data=dataset[["petal_length","petal_width","species"]]		
data=np.apply_along_axis(np.random.permutation,0,data)
useful_data=[]
for i in range(150):
	if data[i][2]=="Iris-setosa" or data[i][2]=="Iris-virginica":
		useful_data.append(data[i])
		
for i in range (100):
	if useful_data[i][2]=="Iris-setosa":
		useful_data[i][2]=1
	else:
		useful_data[i][2]=0		
		
train_data=useful_data[0:75]
test_data=useful_data[75:100]
#-------------------------------------------------------------------
# sigmoid function & derevative of sigmoid function
#-------------------------------------------------------------------
def sigmoid(val):
	return (1/(1+m.exp(-val)))
#-------------------------------------------------------------------	
def d_sigmoid(val):
	return sigmoid(val)*(1-sigmoid(val))
#-------------------------------------------------------------------

#-------------------------------------------------------------------
def train():
	w1=np.random.rand()
	w2=np.random.rand()
	b=np.random.rand()
	h=0.01        #learning rate
	itr=50000    #no of iteration
	costs=[]
	for i in range(itr):
		ri=np.random.randint(len(train_data))
		point=train_data[ri]
		z=w1*point[0]+w2*point[1]+b
		tar=point[2]
		pred=sigmoid(z)
		#----------------------------------
		#error calculating, wheight tuning
		#----------------------------------
		cost=(pred-tar)**2
		dcost_pred=2*(pred-tar)
		dpred_z=d_sigmoid(z)
		dz_w1=point[0]
		dz_w2=point[1]
		dz_b=1
		
		dcost_w1=dcost_pred*dpred_z*dz_w1
		dcost_w2=dcost_pred*dpred_z*dz_w2
		dcost_b=dcost_pred*dpred_z*dz_b
		
		w1=w1-(h*dcost_w1)
		w2=w2-(h*dcost_w2)
		b=b-(h*dcost_b)
		costs.append(cost)
	plt.plot(costs)
	plt.show()
      
train()		
	

