import csv
import random 
import math 
import operator
file="iris.csv"

#############################################

def loaddata(filename,split,trainset=[],testset=[]):
	with open(filename,'r') as csvfile:
		line=csv.reader(csvfile)
		dataset=list(line)
		for x in range(len(dataset)-1):
			for y in range (4):
				dataset[x][y]=float(dataset[x][y])
			if random.random()<split:
				trainset.append(dataset[x])
			else:
				testset.append(dataset[x])	


###################################################

def gap(ob1,ob2,lenth):
	distance=0
	for x in range(lenth):
		distance+=pow((ob1[x]-ob2[x]),2)
	return math.sqrt(distance)
		
####################################################

def getgroup(trainSet,testobj,k):
	distance=[]
	lenth=len(testobj)-1
	for x in range(len(trainSet)):
		ln = gap(testobj,trainSet[x],lenth)
		distance.append((trainSet[x],ln))
	distance.sort(key=operator.itemgetter(1))
	nb=[]
	for x in range(k):
		nb.append(distance[x][0])
	return nb	

####################################################

def voting(nbr):
	vote={}
	for x in range(len(nbr)):
		res=nbr[x][-1]
		if res in vote:
			vote[res]+=1
		else:
			vote[res]=1

	svote=sorted(vote.items(),key=operator.itemgetter(1),reverse=True)
	return svote[0][0]		
		 	
####################################################

def accurate(testset, pred):
	c=0
	for x in range(len(testset)):		
		if testset[x][-1] == pred[x]:
			c+=1
	return (c/float(len(testset)))*100.0		

#####################################################

def main():
	train=[]
	test=[]
	pred=[]	
	split=0.67
	k=3
	loaddata(file,split,train,test)
	print('train:'+repr(len(train)))
	print('test:'+repr(len(test)))
	for x in range(len(test)):
		nb=getgroup(train,test[x],k)
		res=voting(nb)
		pred.append(res)
		print('> pred:'+repr(res)+'>act:'+repr(test[x][-1]))
	ac=accurate(test,pred)
	print('accurcy:'+repr(ac)+'%')


main()
#####################################################