from Data8klevel2 import dataloader as dd
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

dict_file = './Data8klevel2/vocab.txt'
itokens, otokens = dd.MakeS2SDict(dict_file)
Xtrain, Ytrain = dd.MakeS2SData('./Data8klevel2/trainsrc.txt',
                                './Data8klevel2/traintgt.txt',
                                itokens, otokens,
                                h5_file='./Data8klevel2/train_en2de.h5')
Xvalid, Yvalid = dd.MakeS2SData('./Data8klevel2/valsrc.txt',
                                './Data8klevel2/valtgt.txt',
                                itokens, otokens,
                                h5_file='./Data8klevel2/val_en2de.h5')
Xtest, Ytest = dd.MakeS2SData('./Data8klevel2/testsrc.txt',
                                './Data8klevel2/testtgt.txt',
                                itokens, otokens,
                                h5_file='./Data8klevel2/test_en2de.h5')


print('seq 1 words:', itokens.num())
print('seq 2 words:', otokens.num())
print('train shapes:', Xtrain.shape, Ytrain.shape)
print('valid shapes:', Xvalid.shape, Yvalid.shape)

from transformer import Transformer, LRSchedulerPerStep

d_model = 256
s2s = Transformer(itokens, otokens, len_limit=70, d_model=d_model, d_inner_hid=512, \
                  n_head=8, layers=2, dropout=0.1)

mfile = './models/8kmodel.h5'

lr_scheduler = LRSchedulerPerStep(d_model, 4000)
model_saver = ModelCheckpoint(mfile, save_best_only=True, save_weights_only=True)


s2s.compile(Adam(0.001, 0.9, 0.98, epsilon=1e-9))

try:
    s2s.model.load_weights(mfile)
except:
    print('\n\nnew model')
#for i in range(16):
   # s2s_i.permodel[i].set_weights(s2s.model.get_weights())

print("###########################"
      "###########################")

X = []
with open("./Data8klevel2/testsrc.txt", "r") as fsrc:
    line = fsrc.readline()
    while (line != ""):
        X.append(line)
        line = fsrc.readline()
Y = []
with open('./Data8klevel2/testtgt.txt', 'r') as ftgt:
    line = ftgt.readline()
    while (line != ""):
        Y.append(line)
        line = ftgt.readline()
en = [x.split() for x in X]
'''
rets = []
for i in range(3):
    rets.append(s2s.decode_sequence(en[i], delimiter=' '))
acc = []
with open ('./Data8klevel2/gen_accu_maxdecode.txt', 'w') as fgen:
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
'''
methods = ["input_t_gradient"]
kwargs = [{}]
analyzers = []

'''
X_train_emb = s2s.source_emb_model.predict([Xtrain])
X_train_pos_emb = s2s.source_emb_pos_model.predict([Xtrain])
Y_train_emb = s2s.target_emb_model.predict([Ytrain])
Y_trian_pos_emb = s2s.target_emb_pos_model.predict([Ytrain])
'''
for method, kws in zip(methods, kwargs):
    an = []
    for j in range(16):
        #model = innvestigate.utils.model_wo_softmax(s2s.model)
        #plot_model(s2s.permodel[j],to_file="./modelpng/permodel_%d.png"%j,show_layer_names=True,show_shapes=True)
        analyzer = innvestigate.create_analyzer(method, s2s.permodel[j], **kws)
        analyzer.fit([X_train_emb,X_train_pos_emb,Y_train_emb,Y_trian_pos_emb], batch_size=64, verbose=1)
        an.append(analyzer)
    analyzers.append(an)

test_sample_indices = [97, 175, 300, 686, 754, 543]
# a variable to store analysis results.
maxlen = 17
analysis = np.zeros([len(test_sample_indices), len(analyzers), 1, maxlen])
for i, ridx in enumerate(test_sample_indices):
    source_seq = [Xtest[ridx]]
    source_seq_emb = s2s.i_word_emb(source_seq)
    source_seq_pos_emb = s2s.pos_emb(source_seq)
    decoded_tokens = []
    target_seq = np.zeros((1, maxlen), dtype='int32')
    target_seq[0, 0] = s2s.o_tokens.startid()
    target_seq_emb = s2s.o_word_emb(target_seq)
    target_seq_pos_emb = s2s.pos_emb(target_seq)
    for j in range(s2s.len_limit - 1):
        output = s2s.permodel[j].predict_on_batch([source_seq_emb,source_seq_pos_emb,target_seq_emb,target_seq_pos_emb])
        for aidx, analyzer in enumerate(analyzers):
            wwwww = analyzer[j].analyze([source_seq_emb,source_seq_pos_emb,target_seq_emb,target_seq_pos_emb])
            a = analyzer[j].analyze([source_seq_emb,source_seq_pos_emb,target_seq_emb,target_seq_pos_emb])
            #a = np.sum(a, axis=1)
            print(a)
            print("------------------------------------------------------------------")
            #analysis[i, aidx] = a
        sampled_index = np.argmax(output[0])
        sampled_token = s2s.o_tokens.token(sampled_index)
        decoded_tokens.append(sampled_token)
        if sampled_index == s2s.o_tokens.endid(): break
        target_seq[0, j + 1] = sampled_index
    print(decoded_tokens)

