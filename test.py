import pandas as pd
import numpy as np
#for matplotlib
#import matplotlib.pyplot as plt

headers = ["symboling","normalized-losses","make","fule-type","aspiration","num-of-doors","body-style","drive-wheels","engine-location","wheel-base","length","width","height","curb-weight","engine-type","num-of-cylinders","engine-size","fule-system","bore","stroke","compression-ratio","horsepower","peak-rpm","city-mpg","highway-mpg","price"]

dt = pd.read_csv("data.csv",header = None)
dt.columns=headers
print(dt.head())

#df.dtypes return all the types of data which is stored in the csv file
#print(dt.dtypes)

#df.describe(include="all") will give all the columns present in the data

#df.info will give thw top 30 and bottom 30 data of the file 

#if there is any nan value in the data then we should have to replace it or remove it
#for remove it use
#df.dropna(subset=["coloumn name"],axis=0,inplace=True)
#here axis is used to remove coloumn if there is axis=1 then it will remove the whole column
#and inplace is required to update the data if inplcae is not there then data will not update


#we can replace value as follows
#in below code nan value of normalized-losses replace by mean value of normalized-losses


#convert ? into 0
dt.replace('?',int(0),inplace=True)
print(dt.head())

#now we replace 0 value with the mean value of the column
#mean = dt['normalized-losses'].mean()
#dt['normalized-losses'].replace(0,mean)


#there is a coloum in the table city-mpg we can convert it into city-L/100km
#for conversion miles per galion to kilometer per liter we have to divide from 235
dt['city-mpg'] = 235/dt['city-mpg'] 
#now we rename city-mpg into city-l/100km
dt.rename(columns={'city-mpg':'city-L/100km'},inplace = True)
print(dt['city-L/100km'].head())

#city-L/100km column is of object type we have to convet it into int 

dt["city-L/100km"] = dt["city-L/100km"].astype(int)
print(dt["city-L/100km"].head())

#price is the object type we convert int
dt['price']=dt['price'].astype(int)


#if there is any variation in the columns the we have to normalized it 
#main there are three ways to normalized
#1(simple feature scaling)- dt['length']=dt['lenght']/dt['length'].max()
#2(min-max)- dt['lenght'] = (dt['length']-dt['length'].min())/(dt['length'].max()-dt['length'].min())
#3(z-score)- dt['length']=(df['length']-dt['length'].mean())/dt['length'].std()
#here std() is stander devation

#we can binnig the price which means we can catagories into groups
#let suppose we convert price into 3 bin low, medium,high

binwidth = int((max(dt['price'])-min(dt['price']))/4)
#here we divided it by 4 because bin take one extra space so we divied it by 4 instead of 3		
#now we create an array
bins = range(min(dt['price']), max(dt['price']), binwidth)
#now we have to create labels for each bins
group_names = ['Low','Medium','High']
#now we create a new columns 
dt['price-binned']=pd.cut(dt['price'],bins,labels=group_names)
print(dt['price-binned'])



#one-hot coding
#in this coloum fule have two values one is gas and another one is diesel this will make a dummy 
#which means that where is gas it show 1 and diesel is 0 and where is diesel it show 1 and gas 0
#pd.get_dummies(dt['fuel'])


#we can calculate the individual row 

drive_wheel_counts = dt['drive-wheels'].value_counts()

print(drive_wheel_counts)


#for graph
#for boxplot
#plt.boxplot(x='drive-wheels',y='price',data=dt)

#for scatter type graph
#plt.scatter(x='engine-size',y='price')
#for title the graph
#plt.title('Scatterplot engine-size vs price')
#plt.xlabel('engine-size')
#plt.ylabel('price')











