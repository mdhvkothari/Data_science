import pandas as pd
from pandas import DataFrame , read_csv
name = ['madhav','rahul','rishab','bikesh']
age = [18,16,18,22]
#we can zip these arrya into one
datalist = list(zip(name,age))
print(datalist)
#now we can convert these data into DataFrame
df = pd.DataFrame(data = datalist,columns=['Names','Ages'])
print(df)
#now we can convert this data into csv file
df.to_csv('datafile.csv',index=False)
#to read the above csv file
Location = r'/home/mdhv/data-science/datafile.csv'
df = pd.read_csv(Location)
print(df)
#we can delete this file for this we have to import os
#import os
#os.remove(Location)
#for find the datatype of data
print(df.dtypes)
#for specified column
print(df.Ages.dtypes )
#for shorted data
sorted = df.sort_values(['Ages'],ascending=False)
#.head will give the upper data of the table
print(sorted.head(1))
#another method for finding the max value
print(df['Ages'].max())
