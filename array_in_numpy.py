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


#we can convert 1d array into 2d
array3 = np.arange(10)
array3.shape = 2,5
print(array3)

#we can reshape above array in 1d
print(array3.reshape(1,10))

#create horizontal vector
print(np.r_[0:10:1])

#for vertical vector
print(np.c_[0:10:1])

#we can make the zeros and ones arrya
print(np.ones((5,3)))
#it will convet whole array into int
print(np.zeros((2,3),dtype=np.int32))

#it will make all diagonal one
print(np.identity(3))

#randon number
print(np.random.rand(3,4)*10)

#randon number of integer
print(np.random.randint(30,size=(3,5)))

#we can protect our array not edit it or not modify it

array4 = np.arange(0,10,2)
array4.flags.writeable  = False
print(array4)
# now we can't add or modify array4
