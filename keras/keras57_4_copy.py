import numpy as np
aaa = np.array([1,2,3])
bbb = aaa

bbb[0] = 4
print(bbb)
print(aaa)  #[4,2,3]


print('====================')
ccc = aaa.copy()
ccc[1] = 7
print(ccc)
print(aaa)