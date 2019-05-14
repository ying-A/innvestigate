import numpy as np
import random
datasize = 8000
enlen = 10
delen = enlen
compor = 0
appendlen = int(enlen * compor)
enafterlen = enlen + appendlen
encvocsize = 200
halfevsize = int(0.5 * encvocsize)
ens = np.random.randint(0, encvocsize, size=(datasize, enlen))
des = ens + encvocsize
np.savetxt('trainsrc.txt',ens,fmt='%d')
np.savetxt('traintgt.txt',des,fmt='%d')
