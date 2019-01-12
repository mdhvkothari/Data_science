import numpy as np

array = np.array([[1,2,3],[4,5,6]])
#for sum
print(np.sum(array,axis=1))
print(np.sum(array,axis=0))

#for product

print(np.prod(array,axis=0))
print(np.prod(array,axis=1))

#.clip function it will take two arrguments one is smaller than another one is greater than
#those value is smaller than is will convert into smaller one and larger will convert into
#larger one

print(array.clip(3,4))

array2 = np.array([1.01,2.22,6.75])
print(array2.round())
print(array2.round(decimals=1))

#np.argsort will give the indexing of the sort elements
print(np.argsort(array))
