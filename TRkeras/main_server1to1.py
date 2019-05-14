import sys
sys.path.append("..")
from TRkeras.Data8klevel1 import dataloader as dd
from keras.optimizers import *
from keras.callbacks import *
from keras.utils.vis_utils import plot_model

dict_file = './Data8klevel1/vocab.txt'
itokens, otokens = dd.MakeS2SDict(dict_file)
Xtrain, Ytrain = dd.MakeS2SData('./Data8klevel1/trainsrc.txt',
                                './Data8klevel1/traintgt.txt',
                                itokens, otokens,
                                h5_file='./Data8klevel1/train_en2de.h5')
Xvalid, Yvalid = dd.MakeS2SData('./Data8klevel1/valsrc.txt',
                                './Data8klevel1/valtgt.txt',
                                itokens, otokens,
                                h5_file='./Data8klevel1/val_en2de.h5')
print('seq 1 words:', itokens.num())
print('seq 2 words:', otokens.num())
print('train shapes:', Xtrain.shape, Ytrain.shape)
print('valid shapes:', Xvalid.shape, Yvalid.shape)

#from transformer_with_input_emb import Transformer, LRSchedulerPerStep
from TRkeras.transformer import Transformer, LRSchedulerPerStep

d_model = 256
s2s = Transformer(itokens, otokens, len_limit=70, d_model=d_model, d_inner_hid=512, \
                  n_head=8, layers=2, dropout=0.1)

mfile = './models1to1/ckpt-{epoch:02d}-val_accu_{val_accu:.5f}.h5'

lr_scheduler = LRSchedulerPerStep(d_model, 4000)
model_saver = ModelCheckpoint(mfile, save_weights_only=True,period=1)

s2s.compile(Adam(0.001, 0.9, 0.98, epsilon=1e-9))
try:
    s2s.model.load_weights(mfile)
except:
    print('\n\nnew model')

if 'test' in sys.argv:
    X = []
    with open("./Data8klevel1/testsrc.txt", "r") as fsrc:
        line = fsrc.readline()
        while (line != ""):
            X.append(line)
            line = fsrc.readline()
    Y = []
    with open('./Data8klevel1/testtgt.txt', 'r') as ftgt:
        line = ftgt.readline()
        while (line != ""):
            Y.append(line)
            line = ftgt.readline()
    en = [x.split() for x in X]

    rets = []
    for i in range(3):
        rets.append(s2s.decode_sequence(en[i], delimiter=' '))
    acc = []
    with open ('./Data8klevel1/gen_accu_maxdecode.txt', 'w') as fgen:
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
    s2s.model.summary()
    s2s.model.fit([Xtrain, Ytrain], None, batch_size=50, epochs=40, \
                  validation_data=([Xvalid, Yvalid], None), \
                  callbacks=[lr_scheduler, model_saver])
     # val_accu @ 30 epoch: 0.7045
