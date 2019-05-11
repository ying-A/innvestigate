import random, os, sys
import numpy as np
from keras.models import *
from keras.layers import *
from keras.callbacks import *
from keras.initializers import *
import tensorflow as tf

try:
    from tqdm import tqdm
    from TRkeras.Data8k.dataloader import TokenList, pad_to_longest
# for transformer
except:
    pass


class LayerNormalization(Layer):
    def __init__(self, eps=1e-6, **kwargs):
        self.eps = eps
        super(LayerNormalization, self).__init__(**kwargs)

    def build(self, input_shape):
        self.gamma = self.add_weight(name='gamma', shape=input_shape[-1:], initializer=Ones(), trainable=True)
        self.beta = self.add_weight(name='beta', shape=input_shape[-1:], initializer=Zeros(), trainable=True)
        super(LayerNormalization, self).build(input_shape)

    def call(self, x):
        mean = K.mean(x, axis=-1, keepdims=True)
        std = K.std(x, axis=-1, keepdims=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

    def compute_output_shape(self, input_shape):
        return input_shape


# It's safe to use a 1-d mask for self-attention
class ScaledDotProductAttention():
    def __init__(self, attn_dropout=0.1):
        self.dropout = Dropout(attn_dropout)

    def __call__(self, q, k, v, mask):  # mask_k or mask_qk

        temper = tf.sqrt(tf.cast(tf.shape(k)[-1], dtype='float32'))
        # lambda 11 19 31 39 47 55
        attn = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=[2, 2]) / temper)([q, k])  # shape=(batch, q, k)
        if mask is not None:
            # lambda 12 20 32 40 48 56
            mmask = Lambda(lambda x: (-1e+9) * (1. - K.cast(x, 'float32')))(mask)
            attn = Add()([attn, mmask])

        attn = Activation('softmax')(attn)

        attn = self.dropout(attn)
        # lambda 13 21 33 41 49 57
        output = Lambda(lambda x: K.batch_dot(x[0], x[1]))([attn, v])
        return output, attn


class MultiHeadAttention():
    # mode 0 - big martixes, faster; mode 1 - more clear implementation
    def __init__(self, n_head, d_model, dropout):
        self.n_head = n_head
        self.d_k = self.d_v = d_k = d_v = d_model // n_head
        self.dropout = dropout
        self.qs_layer = Dense(256, use_bias=False)
        self.ks_layer = Dense(256, use_bias=False)
        self.vs_layer = Dense(256, use_bias=False)

        self.attention = ScaledDotProductAttention()
        self.w_o = TimeDistributed(Dense(d_model))

    def __call__(self, q, k, v, mask=None):
        d_k, d_v = self.d_k, self.d_v
        n_head = self.n_head

        qs = self.qs_layer(q)  # [batch_size, len_q, n_head*d_k]
        ks = self.ks_layer(k)
        vs = self.vs_layer(v)

        def reshape1(x):
            s = tf.shape(x)  # [batch_size, len_q, n_head * d_k]
            x = tf.reshape(x, [-1, 16, 8, 256 // 8])
            x = tf.transpose(x, [2, 0, 1, 3])
            x = tf.reshape(x, [-1, 16, 256 // 8])  # [n_head * batch_size, len_q, d_k]
            return x

        # 3lambda :7 8 9 15 16 17 27 28 29 35 36 37 43 44 45 52 52 53
        qs = Lambda(reshape1)(qs)
        ks = Lambda(reshape1)(ks)
        vs = Lambda(reshape1)(vs)



        if mask is not None:
            # lambda 10 18 25 30 38 46 54
            mask = Lambda(lambda x: K.repeat_elements(x, n_head, 0))(mask)
        # head lambda 13 21 25 33 41 49 57
        # attn dp 2 4 6 7 9 10 lamda 25
        head, attn = self.attention(qs, ks, vs, mask=mask)

        def reshape2(x):
            s = tf.shape(x)  # [n_head * batch_size, len_v, d_v]
            x = tf.reshape(x, [8, -1, 16, 32])
            x = tf.transpose(x, [1, 2, 0, 3])
            x = tf.reshape(x, [-1, 16, 256])  # [batch_size, len_v, n_head * d_v]
            return x
        #lambda 14 22 25 34 42 50 58
        head = Lambda(reshape2)(head)
        outputs = self.w_o(head)
        outputs = Dropout(self.dropout)(outputs)
        return outputs, attn

class PositionwiseFeedForward():
    def __init__(self, d_hid, d_inner_hid, dropout=0.1):
        self.w_1 = Conv1D(d_inner_hid, 1, activation='relu')
        self.w_2 = Conv1D(d_hid, 1)
        self.layer_norm = LayerNormalization()
        self.dropout = Dropout(dropout)

    def __call__(self, x):
        output = self.w_1(x)
        output = self.w_2(output)
        output = self.dropout(output)
        output = Add()([output, x])
        return self.layer_norm(output)


class EncoderLayer():
    def __init__(self, d_model, d_inner_hid, n_head, dropout=0.1):
        self.self_att_layer = MultiHeadAttention(n_head, d_model, dropout=dropout)
        self.pos_ffn_layer = PositionwiseFeedForward(d_model, d_inner_hid, dropout=dropout)
        self.norm_layer = LayerNormalization()

    def __call__(self, enc_input, mask=None):
        output, slf_attn = self.self_att_layer(enc_input, enc_input, enc_input, mask=mask)
        output = self.norm_layer(Add()([enc_input, output]))
        output = self.pos_ffn_layer(output)
        return output, slf_attn


class DecoderLayer():
    def __init__(self, d_model, d_inner_hid, n_head, dropout=0.1):
        self.self_att_layer = MultiHeadAttention(n_head, d_model, dropout=dropout)
        self.enc_att_layer = MultiHeadAttention(n_head, d_model, dropout=dropout)
        self.pos_ffn_layer = PositionwiseFeedForward(d_model, d_inner_hid, dropout=dropout)
        self.norm_layer1 = LayerNormalization()
        self.norm_layer2 = LayerNormalization()

    def __call__(self, dec_input, enc_output, self_mask=None, enc_mask=None, dec_last_state=None):
        if dec_last_state is None: dec_last_state = dec_input
        output, slf_attn = self.self_att_layer(dec_input, dec_last_state, dec_last_state, mask=self_mask)
        x = self.norm_layer1(Add()([dec_input, output]))
        output, enc_attn = self.enc_att_layer(x, enc_output, enc_output, mask=enc_mask)
        x = self.norm_layer2(Add()([x, output]))
        output = self.pos_ffn_layer(x)
        return output, slf_attn, enc_attn


def GetPosEncodingMatrix(max_len, d_emb):
    pos_enc = np.array([
        [pos / np.power(10000, 2 * (j // 2) / d_emb) for j in range(d_emb)]
        if pos != 0 else np.zeros(d_emb)
        for pos in range(max_len)
    ])
    pos_enc[1:, 0::2] = np.sin(pos_enc[1:, 0::2])  # dim 2i
    pos_enc[1:, 1::2] = np.cos(pos_enc[1:, 1::2])  # dim 2i+1
    return pos_enc


def GetPadMask(q, k):
    '''
    shape: [B, Q, K]
    '''
    ones = K.expand_dims(K.ones_like(q, 'float32'), -1)
    mask = K.cast(K.expand_dims(K.not_equal(k, 0), 1), 'float32')
    mask = K.batch_dot(ones, mask, axes=[2, 1])
    return mask


def GetSubMask(s):
    '''
    shape: [B, Q, K], lower triangle because the i-th row should have i 1s.
    '''
    len_s = 16
    bs = K.shape(s)[0]
    mask = K.cumsum(tf.eye(len_s, batch_shape=[bs]), 1)
    return mask


class SelfAttention():
    def __init__(self, d_model, d_inner_hid, n_head, layers=6, dropout=0.1):
        self.layers = [EncoderLayer(d_model, d_inner_hid, n_head, dropout) for _ in range(layers)]

    def __call__(self, src_emb, src_seq, return_att=False, active_layers=999):
        if return_att: atts = []
        # lambda 6
        mask = Lambda(lambda x: K.cast(K.greater(x, 0), 'float32'))(src_seq)
        x = src_emb
        for enc_layer in self.layers[:active_layers]:
            x, att = enc_layer(x, mask)
            if return_att: atts.append(att)
        return (x, atts) if return_att else x


class Decoder():
    def __init__(self, d_model, d_inner_hid, n_head, layers=6, dropout=0.1):
        self.layers = [DecoderLayer(d_model, d_inner_hid, n_head, dropout) for _ in range(layers)]

    def __call__(self, tgt_emb, tgt_seq, src_seq, enc_output, return_att=False, active_layers=999):
        x = tgt_emb
        # lambda 14 22 23 34 42 50 58
        self_pad_mask = Lambda(lambda x: GetPadMask(x, x))(tgt_seq)
        # lambda 14 22 24 34 42 50 58    ------------->none 24
        self_sub_mask = Lambda(GetSubMask)(tgt_seq)
        #lambda 25
        self_mask = Lambda(lambda x: K.minimum(x[0], x[1]))([self_pad_mask, self_sub_mask])
        #lambda 26
        enc_mask = Lambda(lambda x: GetPadMask(x[0], x[1]))([tgt_seq, src_seq])

        if return_att: self_atts, enc_atts = [], []
        for dec_layer in self.layers[:active_layers]:
            x, self_att, enc_att = dec_layer(x, enc_output, self_mask, enc_mask)
            if return_att:
                self_atts.append(self_att)
                enc_atts.append(enc_att)
        return (x, self_atts, enc_atts) if return_att else x


class Transformer:
    def __init__(self, i_tokens, o_tokens, len_limit, d_model=256, \
                 d_inner_hid=512, n_head=4, layers=2, dropout=0.1, \
                 share_word_emb=False):
        self.i_tokens = i_tokens
        self.o_tokens = o_tokens
        self.len_limit = len_limit
        self.d_model = d_model
        self.decode_model = None
        self.readout_model = None
        self.layers = layers
        d_emb = d_model

        self.src_loc_info = True

        d_k = d_v = d_model // n_head
        assert d_k * n_head == d_model and d_v == d_k

        self.pos_emb = PosEncodingLayer(len_limit, d_emb) if self.src_loc_info else None

        self.emb_dropout = Dropout(dropout)

        self.i_word_emb = Embedding(i_tokens.num(), d_emb)
        if share_word_emb:
            assert i_tokens.num() == o_tokens.num()
            self.o_word_emb = self.i_word_emb
        else:
            self.o_word_emb = Embedding(o_tokens.num(), d_emb)

        self.encoder = SelfAttention(d_model, d_inner_hid, n_head, layers, dropout)
        self.decoder = Decoder(d_model, d_inner_hid, n_head, layers, dropout)
        #self.target_layer = TimeDistributed(Dense(o_tokens.num(),input_shape=self.decoder.output_shape,use_bias=False))
        self.target_layer = TimeDistributed(Dense(o_tokens.num(), use_bias=False))


    def compile(self, optimizer='adam', active_layers=999):
        src_seq_input = Input(shape=(16,), dtype='int32')
        tgt_seq_input = Input(shape=(17,), dtype='int32')
        src_seq = src_seq_input
        #lambda2
        tgt_seq = Lambda(lambda x: x[:, :-1])(tgt_seq_input)
        #lambda3
        tgt_true = Lambda(lambda x: x[:, 1:])(tgt_seq_input)
        #print(tgt_true)
        src_emb = self.i_word_emb(src_seq)
        self.src_emb = src_emb

        #print(src_emb)
        tgt_emb = self.o_word_emb(tgt_seq)
        self.tgt_emb = tgt_emb
        '''
        self.source_emb_model = Model([src_seq_input], src_emb)
        self.source_emb_pos_model = Model([src_seq_input], self.pos_emb(src_seq))
        self.target_emb_model = Model([ tgt_seq_input], tgt_emb)
        self.target_emb_pos_model = Model([ tgt_seq_input], self.pos_emb(tgt_seq))
        '''

        #print(self.pos_emb(src_seq))
        if self.pos_emb:
            self.src_pos_emb = self.pos_emb(src_seq)
            self.tgt_pos_emb = self.pos_emb(tgt_seq)
            #lambda1 16,256
            src_emb = add_layer([src_emb, self.src_pos_emb])
            #lambda1 11,256
            tgt_emb = add_layer([tgt_emb, self.tgt_pos_emb])



        #print(src_emb)
        #drop1 16,256
        src_emb = self.emb_dropout(src_emb)
        #ln3 16,256
        enc_output = self.encoder(src_emb, src_seq, active_layers=active_layers)
        #ln8 11,256
        dec_output = self.decoder(tgt_emb, tgt_seq, src_seq, enc_output, active_layers=active_layers)
        final_output = self.target_layer(dec_output)

        def get_slice(x, index):
            return x[:, index,:]
        self.permodel = []
        for i in range (16):
            per_layer = Lambda(get_slice,arguments={'index': i})(final_output)
            self.permodel.append(Model([src_seq_input, tgt_seq_input],per_layer))



        def get_loss(y_pred, y_true):
            y_true = tf.cast(y_true, 'int32')
            #print(y_true)
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)
            mask = tf.cast(tf.not_equal(y_true, 0), 'float32')
            loss = tf.reduce_sum(loss * mask, -1) / tf.reduce_sum(mask, -1)
            loss = K.mean(loss)
            return loss

        def get_accu(y_pred, y_true):
            mask = tf.cast(tf.not_equal(y_true, 0), 'float32')
            corr = K.cast(K.equal(K.cast(y_true, 'int32'), K.cast(K.argmax(y_pred, axis=-1), 'int32')), 'float32')
            corr = K.sum(corr * mask, -1) / K.sum(mask, -1)
            return K.mean(corr)

        loss = get_loss(final_output, tgt_true)
        self.ppl = K.exp(loss)
        self.accu = get_accu(final_output, tgt_true)

        self.model = Model([src_seq_input, tgt_seq_input], final_output)
        self.model.add_loss([loss])

        self.model.compile(optimizer, None)
        self.model.metrics_names.append('ppl')
        self.model.metrics_tensors.append(self.ppl)
        self.model.metrics_names.append('accu')
        self.model.metrics_tensors.append(self.accu)

    def make_src_seq_matrix(self, input_seq):
        src_seq = np.zeros((1, len(input_seq) + 2), dtype='int32')
        src_seq[0, 0] = self.i_tokens.startid()
        for i, z in enumerate(input_seq): src_seq[0, 1 + i] = self.i_tokens.id(z)
        src_seq[0, len(input_seq) + 1] = self.i_tokens.endid()
        return src_seq

    def decode_sequence(self, input_seq, delimiter=''):
        src_seq = self.make_src_seq_matrix(input_seq)
        decoded_tokens = []
        target_seq = np.zeros((1, self.len_limit), dtype='int32')
        target_seq[0, 0] = self.o_tokens.startid()
        for i in range(self.len_limit - 1):
            output = self.model.predict_on_batch([src_seq, target_seq])
            sampled_index = np.argmax(output[0, i, :])
            sampled_token = self.o_tokens.token(sampled_index)
            decoded_tokens.append(sampled_token)
            if sampled_index == self.o_tokens.endid(): break
            target_seq[0, i + 1] = sampled_index
        return delimiter.join(decoded_tokens[:-1])


class PosEncodingLayer:
    def __init__(self, max_len, d_emb):
        self.pos_emb_matrix = Embedding(max_len, d_emb, trainable=False, \
                                        weights=[GetPosEncodingMatrix(max_len, d_emb)])

    def get_pos_seq(self, x):
        mask = K.cast(K.not_equal(x, 0), 'int32')
        pos = K.cumsum(K.ones_like(x, 'int32'), 1)
        return pos * mask

    def __call__(self, seq, pos_input=False):
        x = seq
        if not pos_input: x = Lambda(self.get_pos_seq)(x)
        return self.pos_emb_matrix(x)


class AddPosEncoding:
    def __call__(self, x):
        _, max_len, d_emb = K.int_shape(x)
        pos = GetPosEncodingMatrix(max_len, d_emb)
        x = Lambda(lambda x: x + pos)(x)
        return x


class LRSchedulerPerStep(Callback):
    def __init__(self, d_model, warmup=4000):
        self.basic = d_model ** -0.5
        self.warm = warmup ** -1.5
        self.step_num = 0

    def on_batch_begin(self, batch, logs=None):
        self.step_num += 1
        lr = self.basic * min(self.step_num ** -0.5, self.step_num * self.warm)
        K.set_value(self.model.optimizer.lr, lr)


add_layer = Lambda(lambda x: x[0] + x[1], output_shape=lambda x: x[0])
# use this because keras may get wrong shapes with Add()([])

if __name__ == '__main__':
    print('done')
