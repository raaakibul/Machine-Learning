import numpy as np
d1 = np.array(([1,2,3]))
print(d1)
print(d1.shape)
d2 = np.array(([1,2,3],[4,5,6]))
print(d2)
print(d2.shape)
d3 = np.array([[[1,2,3],[4,5,6]],[[7,8,9],[11,22,33]]])
print(d3)
print(d3.shape)

print(np.array([range(1,5), range(1,5)]))
print(np.arange(0,10,2))

print(np.zeros((2,2)))
print(np.ones((2,2), dtype='int'))