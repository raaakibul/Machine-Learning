import numpy as np 

arr= np.array([[[1,2,3],[4,5,6]],[[7,8,9],[11,22,33]]])
print(arr)
print(arr.shape)
print(arr[1,1])

print(arr[0][1,2])
print(arr[0,1,2])

arr = np.array(([1,2,3],[4,5,6]))
print(arr)
print(arr[0,2])
print(arr[1,1])

# NaN= not a number and INF = infinity
nums = np.array([1,2,np.inf], dtype='float')
print(nums)
print(np.isinf(nums))

print(sum(np.isinf(nums)))