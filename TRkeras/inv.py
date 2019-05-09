from Data8k import dataloader as dd
from keras.optimizers import *
from keras.callbacks import *
import innvestigate

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

print("###########################"
      "###########################")

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
'''
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
'''
methods = ["input_t_gradient"]
kwargs = [{}]
analyzers = []

for method, kws in zip(methods, kwargs):
    an = []
    for j in range(16):
        plot_model(s2s.permodel[j],to_file="./modelpng/permodel_%d.png"%j,show_layer_names=True,show_shapes=True)
        analyzer = innvestigate.create_analyzer(method, s2s.permodel[j], **kws)
        analyzer.fit([Xtrain,Ytrain], batch_size=64, verbose=1)
        an.append(analyzer)
    analyzers.append(an)

test_sample_indices = [97, 175, 300, 686, 754, 543]
# a variable to store analysis results.
maxlen = 17
analysis = np.zeros([len(test_sample_indices), len(analyzers), 1, maxlen])
for i, ridx in enumerate(test_sample_indices):
    source_seq = [Xtest[ridx]]
    decoded_tokens = []
    target_seq = np.zeros((1, maxlen), dtype='int32')
    target_seq[0, 0] = s2s.o_tokens.startid()
    for j in range(s2s.len_limit - 1):
        output = s2s.permodel[j].predict_on_batch([source_seq,target_seq])
        for aidx, analyzer in enumerate(analyzers):
            wwwww = analyzer[j].analyze([source_seq,target_seq])
            a = analyzer[j].analyze([source_seq,target_seq])
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

