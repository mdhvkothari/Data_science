import numpy  as np
list = np.array([1,2,3,4,5,6])
#this is not possible in simple list in python
list+=1
print(list)
#.shape will give the lenght as well as dimenssion
print(list.shape)
#.ndim give the dimenssion of the array
print(list.ndim)

matrix = np.array([[1,2,3],
                   [4,5,6],
                   [7,8,9]])
print(matrix.shape)
print(matrix.ndim)
#how to indexing the array
print(matrix[1,2])
#this will copy matrix into matrix1 and if we update matrix1 it will not effect the matrix
#if we don't use copy then if we update matrix1 then it will change matrix as well
matrix1 = matrix.copy()

list1 = np.arange(0,100,20)

#conversion
#it will convert all int into float
list2 = np.array([1,2,3], dtype = np.float32)
print(list2)

#if there is any string the array then all items will convert into string and if any float is there
#all elements convert into float and if there is both string and float is there then
#all will convert into string string have more piority then float then int
#if we would not convert into any datatype then use below
list3 = np.array([1,"2",3.0], dtype = np.object)
print(list3)
