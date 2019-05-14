import os, sys, time, random
sys.path.append("..")
import ljqpy
import h5py
import numpy as np

class TokenList:
	def __init__(self, token_list):
		self.id2t = ['<PAD>', '<UNK>', '<S>', '</S>'] + token_list
		self.t2id = {v:k for k,v in enumerate(self.id2t)}
	def id(self, x):	return self.t2id.get(x, 1)
	def token(self, x):	return self.id2t[x]
	def num(self):		return len(self.id2t)
	def startid(self):  return 2
	def endid(self):    return 3
	
def pad_to_longest(xs, tokens, max_len=999):
	longest = min(len(max(xs, key=len))+2, max_len)
	X = np.zeros((len(xs), longest), dtype='int32')
	X[:,0] = tokens.startid()
	for i, x in enumerate(xs):
		x = x[:max_len-2]
		for j, z in enumerate(x):
			X[i,1+j] = tokens.id(z)
		X[i,1+len(x)] = tokens.endid()
	print(X[0])
	print(X[1])
	return X

def pad_to_length(xs, tokens, length = 999):
	longest = length
	X = np.zeros((len(xs), longest), dtype='int32')
	X[:,0] = tokens.startid()
	for i, x in enumerate(xs):
		x = x[:12-2]
		for j, z in enumerate(x):
			X[i,1+j] = tokens.id(z)
		X[i,1+len(x)] = tokens.endid()
	#print(X[0])
	#print(X[1])
	return X


def MakeS2SDict( dict_file=None):
	if dict_file is not None and os.path.exists(dict_file):
		print('loading', dict_file)
		lst = ljqpy.LoadList(dict_file)
		midpos = lst.index('<@@@>')
		itokens = TokenList(lst[:midpos])
		otokens = TokenList(lst[midpos+1:])
		return itokens, otokens

def MakeS2SData(source_path,target_path, itokens=None, otokens=None, h5_file=None, max_len=18):
	if h5_file is not None and os.path.exists(h5_file):
		print('loading', h5_file)
		with h5py.File(h5_file) as dfile:
			X, Y = dfile['X'][:], dfile['Y'][:]
		return X, Y
	Xs = [[], []]
	with open(source_path,'r') as fsrc:
		line = fsrc.readline()
		while(line!=""):
			Xs[0].append(line.split())
			line = fsrc.readline()
	with open(target_path,'r') as ftgt:
		line = ftgt.readline()
		while(line!=""):
			Xs[1].append(line.split())
			line = ftgt.readline()
	print("DONE")
	X, Y = pad_to_length(Xs[0], itokens, 16), pad_to_length(Xs[1], otokens, 17)
	print(X[0])
	print(Y[0])
	if h5_file is not None:
		with h5py.File(h5_file, 'w') as dfile:
			dfile.create_dataset('X', data=X)
			dfile.create_dataset('Y', data=Y)
	return X, Y


def S2SDataGenerator(fn, itokens, otokens, batch_size=64, delimiter=' ', max_len=999):
	Xs = [[], []]
	while True:
		for ss in ljqpy.LoadCSVg(fn):
			for seq, xs in zip(ss, Xs):
				xs.append(list(seq.split(delimiter)))
			if len(Xs[0]) >= batch_size:
				X, Y = pad_to_longest(Xs[0], itokens, max_len), pad_to_longest(Xs[1], otokens, max_len)
				yield [X, Y], None
				Xs = [[], []]

if __name__ == '__main__':
	dict_file = 'C:/Users/trio/innvestigate/TRkeras/Data8klevel1/vocab.txt'
	itokens, otokens = MakeS2SDict(dict_file)
	X, Y = MakeS2SData('C:/Users/trio/innvestigate/TRkeras/Data8klevel1/trainsrc.txt',
					   'C:/Users/trio/innvestigate/TRkeras/Data8klevel1/traintgt.txt',
					   itokens, otokens,
					   h5_file='C:/Users/trio/innvestigate/TRkeras/Data8klevel1/train_en2de.h5')
	print(X.shape, Y.shape)
