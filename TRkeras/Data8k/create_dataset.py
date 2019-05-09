import numpy as np
import random
datasize = 1000
enlen = 10
delen = enlen
compor = 0.4
appendlen = int(enlen * compor)
enafterlen = enlen + appendlen
encvocsize = 200
halfevsize = int(0.5 * encvocsize)
ens = np.random.randint(0, encvocsize, size=(datasize, enlen))
des = ens + encvocsize
afterens = np.zeros((datasize, enafterlen), dtype=int)
afterdes = np.zeros((datasize, delen), dtype=int)
for pairidx in range(datasize):
    en = ens[pairidx]
    de = des[pairidx]
    #print(en)
    #print(de)
    idx = random.sample(range(0, enlen), appendlen)
    sortidx = np.sort(idx)
    #print(sortidx)
    for j in sortidx:
        if en[j] < halfevsize:
            de[j] = en[j] + 2 * encvocsize
        if en[j] >= halfevsize:
            de[j] = en[j] - halfevsize + 2 * encvocsize
    addidx = np.arange(0, appendlen)
    finalidx = sortidx + addidx
    for i in finalidx:
        if en[i] < halfevsize:
            en = np.insert(en, i+1, en[i] + halfevsize)
        if en[i] >= halfevsize:
            en = np.insert(en, i, en[i] - halfevsize)
    afterens[pairidx] = en
    afterdes[pairidx] = de
    #print(afterens[pairidx])
    #print(afterdes[pairidx])
np.savetxt('testsrc.txt',afterens,fmt='%d')
np.savetxt('testtgt.txt',afterdes,fmt='%d')
