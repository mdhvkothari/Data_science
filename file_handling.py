import numpy as np

array = np.array([1,2,3,45])
#here fmd is used to store in unit digit otherwise is will store in decimals digits
np.savetxt("file.txt",array,fmt="%d")

#we can save our file in different format like .npy it will convert data into different format
#for using .npy file we load into the program
array1 = np.array([4,5,6,78])
np.save("file1.npy",array1)
#for using npy file we first load file1
array2  = np.load("file1.npy")
print(array2)

#we can convert file into zip like format with the help of   .npz
array3 = np.arange(100)
#it will create a zip file and in this file we find two .npy format file which contain array3 and array1
np.savez("file2.npz",a=array3,b=array2)
