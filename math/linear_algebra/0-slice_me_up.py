#!/usr/bin/env python3
arr = [9, 8, 2, 3, 9, 4, 1, 0, 3]
arr1 =  slice(0, 2)
arr2 =  slice(4,None)
arr3 =  slice(1, 6)
print("The first two numbers of the array are: {}".format(arr[arr1]))
print("The last five numbers of the array are: {}".format(arr[arr2]))
print("The 2nd through 6th numbers of the array are: {}".format(arr[arr3]))
