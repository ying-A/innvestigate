from Data8k import dataloader as dd
from keras.optimizers import *
from keras.callbacks import *
import random, os, sys
import numpy as np
from keras.models import *
from keras.layers import *
from keras.callbacks import *
from keras.initializers import *
import tensorflow as tf
import innvestigate
import innvestigate.utils as iutils

import numpy as np
from keras.utils.vis_utils import plot_model

dict_file = './Data8k/vocab.txt'
itokens, otokens = dd.MakeS2SDict(dict_file)
Xtrain, Ytrain = dd.MakeS2SData('./Data8k/trainsrc.txt',
                                './Data8k/traintgt.txt',
                                itokens, otokens,
                                h5_file='./Data8k/train_en2de.h5')
Xvalid, Yvalid = dd.MakeS2SData('./Data8k/valsrc.txt',
                                './Data8k/valtgt.txt',
                                itokens, otokens,
                                h5_file='./Data8k/val_en2de.h5')
Xtest, Ytest = dd.MakeS2SData('./Data8k/testsrc.txt',
                                './Data8k/testtgt.txt',
                                itokens, otokens,
                                h5_file='./Data8k/test_en2de.h5')

from transformer import Transformer, LRSchedulerPerStep
from transformer_with_input_emb import Transformer_i,LRSchedulerPerStep_i

d_model = 256
s2s = Transformer(itokens, otokens, len_limit=70, d_model=d_model, d_inner_hid=512, \
                  n_head=8, layers=2, dropout=0.1)
s2s_i = Transformer_i(itokens, otokens, len_limit=70, d_model=d_model, d_inner_hid=512, \
                  n_head=8, layers=2, dropout=0.1)

mfile = './models/8kmodel.h5'
mfile_i = '/models/8kmodel_i.h5'

lr_scheduler_i = LRSchedulerPerStep_i(d_model, 4000)
model_saver = ModelCheckpoint(mfile, save_best_only=True, save_weights_only=True)

model_saver_i = ModelCheckpoint(mfile_i, save_best_only=True, save_weights_only=True)
s2s.compile(Adam(0.001, 0.9, 0.98, epsilon=1e-9))
s2s_i.compile(Adam(0.001, 0.9, 0.98, epsilon=1e-9))

X_train_emb = s2s.source_emb_model.predict([Xtrain])
X_train_pos_emb = s2s.source_emb_pos_model.predict([Xtrain])
Y_train_emb = s2s.target_emb_model.predict([Ytrain])
Y_train_pos_emb = s2s.target_emb_pos_model.predict([Ytrain])

try:
    s2s_i.model.load_weights(mfile_i)
except:
    print('\n\nnew model')
if 'test' in sys.argv:
    X = []
    with open("./Data8k/testsrc.txt", "r") as fsrc:
        line = fsrc.readline()
        while (line != ""):
            X.append(line)
            line = fsrc.readline()
    Y = []
    with open('./Data8k/testtgt.txt', 'r') as ftgt:
        line = ftgt.readline()
        while (line != ""):
            Y.append(line)
            line = ftgt.readline()
    en = [x.split() for x in X]

    rets = []
    for i in range(3):
        rets.append(s2s.decode_sequence(en[i], delimiter=' '))
    acc = []
    with open ('./Data8k/gen_accu_maxdecode.txt', 'w') as fgen:
        for i in range(len(rets)):
            pred = rets[i].split()
            true = Y[i].split()
            fgen.write(rets[i]+"\n")
            cnt = .0
            for j in range(len(pred)):
                if true[j] == pred[j]:
                    cnt += 1
            acc.append(cnt/len(pred))
        fgen.write(str(np.mean(acc)))
else:
    s2s_i.model.summary()
    s2s_i.model.fit([Xtrain, Ytrain], None, batch_size=50, epochs=30, \
                  validation_data=([Xvalid, Yvalid], None), \
                  callbacks=[lr_scheduler_i, model_saver_i])

