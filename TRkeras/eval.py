from TRkeras.Data8k import dataloader as dd
from keras.optimizers import *
from keras.callbacks import *
import matplotlib.pyplot as plt
from TRkeras.transformer import Transformer, LRSchedulerPerStep
import innvestigate
import numpy as np
import heapq
import os
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

acclog = open('./Data8k/gardient_valid_log.txt','a')
path = "C:\\Users\\trio\\innvestigate\\TRkeras\\models"
mfiles = []
dirfiles = os.listdir(path)
for x in dirfiles:
   mfiles.append(os.path.join(path,x))
iii = 0
invs_acc = []
ys_pre_acc = []
d_model = 256
s2s = Transformer(itokens, otokens, len_limit=70, d_model=d_model, d_inner_hid=512,n_head=8, layers=2, dropout=0.1)
s2s.compile(Adam(0.001, 0.9, 0.98, epsilon=1e-9))
for mfile in mfiles:
    s2s.model.load_weights(mfile)
    print("###########################"
          "###########################")
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

    maxlen = 17
    all_setences_analysis = []
    all_decoded_sentences = []

    y_pre_right_cnt = 0
    y2x_inv_right_cnt = 0
    #for i in range(len(Xvalid)):



    for i in range(len(Xvalid)):
        source_seq = [Xvalid[i]]
        source_words = [s2s.i_tokens.token(srcword_id) for srcword_id in Xvalid[i]]
        #print("X",source_words)
        decoded_tokens = []
        target_seq = np.zeros((1, maxlen), dtype='int32')
        target_seq[0, 0] = s2s.o_tokens.startid()
        #analysis_allstep = []
        for j in range(13):
            y_true = [s2s.o_tokens.token(tgtword_id) for tgtword_id in Yvalid[i]][1:16]
            y_true_step = y_true[j]
            output = s2s.permodel[j].predict_on_batch([source_seq,target_seq])
            y_pred_step_tokenid = np.argmax(output[0])
            y_pred_step = s2s.o_tokens.token(y_pred_step_tokenid)
            decoded_tokens.append(y_pred_step)

            #print("Y",y_true)
            #print("step",j)
            #print("y_true_step",y_true_step)
            #print("y_pred_step",y_pred_step)

            #analysis_perstep = []

            y2x_inv_step_pred_0 = ""
            y2x_inv_step_pred_1 = ""

            for aidx, analyzer in enumerate(analyzers):
                an2or4 = analyzer[j].analyze([source_seq,target_seq])
                a = an2or4[0]#source
                a = np.squeeze(a)#squeeze
                if a.ndim==2:
                    a = np.sum(a, axis=1)#step j analyzer[aidx]'s analysis
                #print(a)
                if methods[0] in ['gradient','gradient.baseline']:
                    a = abs(a)
                b = a.tolist()
                index_in_source_sentence_0 = list(map(b.index, heapq.nlargest(2, b)))[0]
                index_in_source_sentence_1 = list(map(b.index, heapq.nlargest(2, b)))[1]
                y2x_inv_step_pred_0 = source_words[index_in_source_sentence_0]
                y2x_inv_step_pred_1 = source_words[index_in_source_sentence_1]

                #analysis_perstep.append(a) #step j analyzer[aidx]'s analysis was appended to step j all_analyzers's analysis
                #print("------------------------------------------------------------------")
            #analysis_allstep.append(analysis_perstep)
            if y_pred_step_tokenid == s2s.o_tokens.endid(): break
            if (y_true_step == y_pred_step and y_true_step!="</S>"):
                y_pre_right_cnt += 1
                if(int(y_pred_step)>=200 and int(y_pred_step)<=399):
                    y2x_inv_step_true_0 = int(y_pred_step)-200
                    #print("y2x_inv_step_true_0",y2x_inv_step_true_0)
                   # print("y2x_inv_step_pred_0", y2x_inv_step_pred_0)
                    if y2x_inv_step_pred_0!="<S>" and y2x_inv_step_pred_0!="</S>":
                        y2x_inv_step_pred_0 = int(y2x_inv_step_pred_0)
                        if y2x_inv_step_pred_0==y2x_inv_step_true_0:
                            y2x_inv_right_cnt += 1
                elif (int(y_pred_step)>=400 and int(y_pred_step)<=499):
                    y2x_inv_step_true_0 = int(y_pred_step)-400
                    y2x_inv_step_true_1 = y2x_inv_step_true_0 + 100
                    #print("y2x_inv_step_true_0", y2x_inv_step_true_0)
                    #print("y2x_inv_step_pred_0", y2x_inv_step_pred_0)
                    #print("y2x_inv_step_true_1", y2x_inv_step_true_1)
                    #print("y2x_inv_step_pred_1", y2x_inv_step_pred_1)
                    if y2x_inv_step_pred_0 != "<S>" and y2x_inv_step_pred_0 != "</S>" and y2x_inv_step_pred_1 != "<S>" and y2x_inv_step_pred_1 != "</S>":
                        y2x_inv_step_pred_0 = int(y2x_inv_step_pred_0)
                        y2x_inv_step_pred_1 = int(y2x_inv_step_pred_1)
                        if y2x_inv_step_pred_0==y2x_inv_step_true_0 and y2x_inv_step_pred_1==y2x_inv_step_true_1:
                            y2x_inv_right_cnt += 1
                        elif y2x_inv_step_pred_0==y2x_inv_step_true_1 and y2x_inv_step_pred_1==y2x_inv_step_true_0:
                            y2x_inv_right_cnt += 1
            target_seq[0, j + 1] = y_pred_step_tokenid

        #all_decoded_sentences.append(decoded_tokens)
        #all_setences_analysis.append(analysis_allstep)
    y_pre_acc = y_pre_right_cnt / 100 * 10  # num_sens * sen_len
    y2x_inv_acc = y2x_inv_right_cnt/y_pre_right_cnt
    invs_acc.append(y2x_inv_acc)
    ys_pre_acc.append(y_pre_acc)
    print("epoch", iii)
    print("y_pre_acc", y_pre_acc)
    print("y2x_inv_acc", y2x_inv_acc)
    acclog.write(str(iii)+" "+str(y_pre_acc)+" "+str(y2x_inv_acc)+"\n")
    iii +=1

print(invs_acc)
'''
title = ""
plt.title("Method:Gradient", fontsize=24)
plt.xlabel("epoch", fontsize=14)
plt.ylabel("acc", fontsize=14)
epoch = [i for i in range(40)]
plt.plot(epoch, ys_pre_acc, label="ys_pre_acc")
plt.plot(epoch, invs_acc, label="invs_acc")
plt.legend()
plt.show()
'''

