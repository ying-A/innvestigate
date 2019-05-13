from TRkeras.Data8klevel2 import dataloader as dd
from keras.optimizers import *
from keras.callbacks import *
import matplotlib.pyplot as plt
import innvestigate
import numpy as np
from matplotlib import cm, transforms

def plot_text_heatmap(words, scores, title="", width=10, height=0.2, verbose=0, max_word_per_line=20):
    fig = plt.figure(figsize=(width, height))
    ax = plt.gca()
    ax.set_title(title, loc='left')
    tokens = words
    if verbose > 0:
        print('len words : %d | len scores : %d' % (len(words), len(scores)))
    cmap = plt.cm.ScalarMappable(cmap=cm.get_cmap('bwr'))
    cmap.set_clim(0, 1)
    canvas = ax.figure.canvas
    t = ax.transData
    # normalize scores to the followings:
    # - negative scores in [0, 0.5]
    # - positive scores in (0.5, 1]
    normalized_scores = 0.5 * scores / np.max(np.abs(scores)) + 0.5
    if verbose > 1:
        print('Raw score')
        print(scores)
        print('Normalized score')
        print(normalized_scores)
    # make sure the heatmap doesn't overlap with the title
    loc_y = -0.2
    for i, token in enumerate(tokens):
        *rgb, _ = cmap.to_rgba(normalized_scores[i], bytes=True)
        color = '#%02x%02x%02x' % tuple(rgb)
        text = ax.text(0.0, loc_y, token,
                       bbox={
                           'facecolor': color,
                           'pad': 5.0,
                           'linewidth': 1,
                           'boxstyle': 'round,pad=0.5'
                       }, transform=t)

        text.draw(canvas.get_renderer())
        ex = text.get_window_extent()

        # create a new line if the line exceeds the length
        if (i + 1) % max_word_per_line == 0:
            loc_y = loc_y - 2.5
            t = ax.transData
        else:
            t = transforms.offset_copy(text._transform, x=ex.width + 15, units='dots')

    if verbose == 0:
        ax.axis('off')

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

from TRkeras.transformer import Transformer, LRSchedulerPerStep

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
tensor_to_analyze = [s2s.src_emb,s2s.tgt_emb,s2s.src_pos_emb,s2s.tgt_pos_emb]
'''
methods = ["input_t_gradient",
           "gradient.baseline",
           "gradient",
           "random",
           "lrp.epsilon"]
kwargs = [{'tensor_to_analyze':tensor_to_analyze,'methodtype_input':True},
          {'tensor_to_analyze':tensor_to_analyze,'methodtype_input':False},
          {'tensor_to_analyze':tensor_to_analyze,'methodtype_input':False},
          {'tensor_to_analyze':tensor_to_analyze,'methodtype_input':False},
          {'tensor_to_analyze':tensor_to_analyze,'methodtype_input':False},]
'''
methods = ["gradient"]
kwargs = [{'tensor_to_analyze':tensor_to_analyze,'methodtype_input':False}]
allmethods = [
    # Utility.
    "input",
    "random",

    # Gradient based
    "gradient",
    "gradient.baseline",
    "input_t_gradient",
    "deconvnet",
    "guided_backprop",
    "integrated_gradients",
    "smoothgrad",

    # Relevance based
    "lrp",
    "lrp.z",
    "lrp.z_IB",

    "lrp.epsilon",
    "lrp.epsilon_IB",

    "lrp.w_square",
    "lrp.flat",

    "lrp.alpha_beta",

    "lrp.alpha_2_beta_1",
    "lrp.alpha_2_beta_1_IB",
    "lrp.alpha_1_beta_0",
    "lrp.alpha_1_beta_0_IB",
    "lrp.z_plus",
    "lrp.z_plus_fast",

    "lrp.sequential_preset_a",
    "lrp.sequential_preset_b",
    "lrp.sequential_preset_a_flat",
    "lrp.sequential_preset_b_flat",

    # Deep Taylor
    "deep_taylor",
    "deep_taylor.bounded",

    # DeepLIFT
    #"deep_lift": DeepLIFT,
    "deep_lift.wrapper",

    # Pattern based
    "pattern.net",
    "pattern.attribution",
]


analyzers = []
for method, kws in zip(methods, kwargs):
    an = []
    for j in range(16):
        #plot_model(s2s.permodel[j],to_file="./modelpng/permodel_%d.png"%j,show_layer_names=True,show_shapes=True)
        analyzer = innvestigate.create_analyzer(method, s2s.permodel[j], **kws)
        analyzer.fit([Xtrain,Ytrain], batch_size=64, verbose=1)
        an.append(analyzer)
    analyzers.append(an)

test_sample_indices = [97, 175, 300, 686, 754, 543]
# a variable to store analysis results.
maxlen = 17
#analysis = np.zeros([len(test_sample_indices), len(analyzers), 1, maxlen])
all_setences_analysis = []
all_decoded_sentences = []
for i, ridx in enumerate(test_sample_indices):
    source_seq = [Xtest[ridx]]
    decoded_tokens = []
    target_seq = np.zeros((1, maxlen), dtype='int32')
    target_seq[0, 0] = s2s.o_tokens.startid()
    analysis_allstep = []
    for j in range(16):
        output = s2s.permodel[j].predict_on_batch([source_seq,target_seq])
        analysis_perstep = []
        for aidx, analyzer in enumerate(analyzers):
            an2or4 = analyzer[j].analyze([source_seq,target_seq])
            a = an2or4[0]#source
            a = np.squeeze(a)#squeeze
            if a.ndim==2:
                a = np.sum(a, axis=1)#step j analyzer[aidx]'s analysis
            #print(a)
            analysis_perstep.append(a) #step j analyzer[aidx]'s analysis was appended to step j all_analyzers's analysis
            #print("------------------------------------------------------------------")
        #print(len(analysis_perstep))
        analysis_allstep.append(analysis_perstep)
        sampled_index = np.argmax(output[0])
        sampled_token = s2s.o_tokens.token(sampled_index)
        decoded_tokens.append(sampled_token)
        if sampled_index == s2s.o_tokens.endid(): break
        target_seq[0, j + 1] = sampled_index
    #print(len(analysis_allstep))
    #print(decoded_tokens)
    all_decoded_sentences.append(decoded_tokens)
    all_setences_analysis.append(analysis_allstep)
#print(len(all_setences_analysis))

sentences_nums = len(all_setences_analysis)
sentence_length = len(all_setences_analysis[0])
method_nums = len(all_setences_analysis[0][0])
for i,idx in enumerate(test_sample_indices):
    source_words = [s2s.i_tokens.token(srcword_id) for srcword_id in Xtest[idx]]
    print('Review(id=%d): %s' % (idx, ' '.join(source_words)))
    y_true = [s2s.o_tokens.token(tgtword_id) for tgtword_id in Ytest[idx]][1:10]
    y_pred = all_decoded_sentences[i][0:9]
    print("y_true")
    print(y_true)
    print("y_pred")
    print(y_pred)
    for j in range(sentence_length):
        print("Step: ",j)
        for k, method in enumerate(methods):
            plot_text_heatmap(source_words, all_setences_analysis[i][j][k].reshape(-1), title='Method: %s' % method, verbose=0)
            plt.show()
