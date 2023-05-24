import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import math
import pickle
import time
import random
import operator

from IPython.display import Audio
from tr3L10B16MB import Net


"""
Parameter definition
"""

seed = 0

if seed != None:
    model_name = 'model3L10B16MBR'+str(seed)+'.dat'
else:
    model_name = 'model3L10B16MB.dat'

print (model_name)

input_dim = 1
output_dim = 256
hidden_dim = 256
nb_lstm_layers = 3
nb_epochs = 10
batch_size = 16

print('  input dimension: %d' % input_dim)
print('  hidden dimension: %d' % hidden_dim)
print('  output dimension: %d' % output_dim)
print('  number of LSTM layers: %d' % nb_lstm_layers)
print('  number of epochs: %d' % nb_epochs)
print('  batch size: %d' % batch_size)

with open('/home/alumnos/alumno3/work/TFM/data/speech.pkl','rb') as f:
    x = pickle.load(f)

T = len(x[0])
N = len(x)
print("N muestras/señal: ", T)
print("N señales: ", N)

def dec(x, M=256):
    mu = M - 1
    y = x.astype(np.float32)
    y = 2 * (y / mu) - 1
    x = np.sign(y) * (1.0 / mu) * ((1.0 + mu)**abs(y) - 1.0)
    return x

x[:,0] = 0

#print("Valores mínimos y máximos de la señal tras decimado: " + str(x.min()) + " - " + str(x.max()))

#fs = 8000
#x_manipulate = dec(x)

#plt.plot(x_manipulate[0,:])
#Audio(x_manipulate[0,:],rate=fs)

""" Test data """

x = x.reshape(N, T, input_dim)

if seed != None:
    random.seed(seed)
    random.shuffle(x)
    print("Random seed = ", seed)

x_test = x[:100]
x_dev = x[100:200]
x_train = x[200:]

""" Loading the trained model """


model = Net(input_dim, hidden_dim, output_dim, nb_lstm_layers)

if torch.cuda.is_available():
    if not torch.cuda.is_initialized():
        torch.cuda.init()   
    model = torch.load(model_name, map_location=torch.device('cuda'))
    print("Boolean para indicar que el modelo ESTÁ en CUDA: " + str(next(model.parameters()).is_cuda))
else:
    model = torch.load(model_name, map_location=torch.device('cpu'))
    print("Boolean para indicar que el modelo NO ESTÁ en CUDA: " + str(next(model.parameters()).is_cuda))

print("CUDA available: " + str(torch.cuda.is_available()))
print("CUDA initialized: " + str(torch.cuda.is_initialized()))

print(model)

#model = model.to(device=torch.device('cuda'))
#model.load_state_dict(torch.load('model1.dat', map_location=torch.device('cuda')))
#model = torch.load('model1.dat', map_location=torch.device('cpu'))
#model = torch.load('model1.dat')
#model = torch.load('/home/alumnos/alumno3/work/TFM/experimento2capas/model.dat')
#model.cuda()

"""
Clase que proporciona los métodos necesarios para codificar un mensaje 
utilizando el método Huffman.
"""

class HuffTree(object):
    def __init__(self, w, symbol=None, zero=None, one=None):
        self.w = w
        self.symbol = symbol #symbol only ever populated on leafs
        self.zero = zero
        self.one = one
    
def _combine(tree1, tree2):
    return HuffTree(w = tree1.w+tree2.w, zero=tree1, one=tree2)

def make_huffman_tree(w_):
    return _build([HuffTree(w, i) for i, w in w_] )
    #return _build([HuffTree(w, i) for i, w in enumerate(w_)] )

def _build(nodes):
    if len(nodes) == 1:
        #bottom out if only one node left
        return nodes[0]
    else:
        #otherwise sort the list, take the bottom two elements and combine
        nodes.sort(key = lambda x: x.w, reverse=True)
        n1 = nodes.pop()
        n2 = nodes.pop()
        nodes.append(_combine(n2, n1))
        return _build(nodes)


def flatten_to_dict(tree, codeword=[], code_dict={}):
    if not tree.symbol is None:
        if len(codeword) == 0:
            codeword.append(0)
        code_dict[tree.symbol] = codeword
    else:
        flatten_to_dict(tree.zero, codeword + [0,], code_dict)
        flatten_to_dict(tree.one, codeword + [1,], code_dict)
    return code_dict

def huffman_encode(m_, d):
    return sum([d[m] for m in m_],[])

def huffman_decode(code_message, hufftree):
    decoded=[]
    t = hufftree
    for i in code_message:
        if i == 0:
            t = t.zero
        elif i == 1:
            t = t.one
        else:
            raise Exception("Code_message not binary")

        if not t.symbol is None:
            decoded.append(t.symbol)
            t = hufftree

    return decoded

from math import log2
"""
Clase que permite codificar y decodificar el mensaje, además de comprobar que 
el mensaje original con el decodificado coinciden.
"""

class CodDecod(object):
  def __init__(self, x_true, x_t0, symbols, model):
        if next(model.parameters()).is_cuda:
            self.x_true = x_true.cuda()
            self.x_t0 = x_t0.cuda()
            self.is_cuda = True
        else:
            self.x_true = x_true
            self.x_t0 = x_t0
            self.is_cuda = False
        self.symbols = symbols
        self.model = model

  def Code(self):
    s = None
    message_encoded = []
    x_t = self.x_t0
    for t in range(len(self.x_true[0])):
        p, s = self.model.predict(x_t, s)
        if self.is_cuda:
            x_t = (torch.FloatTensor([[[self.x_true[0,t]]]])).cuda()
        else:
            x_t = torch.FloatTensor([[[self.x_true[0,t]]]])
        w_ = p.reshape((len(p[0,0]), -1))
        w_ = w_.tolist()
        dictionary = [ (self.symbols[j], w_[j]) for j in range(len(w_)) ]
        h = make_huffman_tree(dictionary)
        
        d = flatten_to_dict(h)
        w = [int(self.x_true[0, t])]
        u = huffman_encode(w, d)
        message_encoded.append(u)
        """
        dict_len= {key: len(value) for key, value in d.items()}
        sorted_key_list = sorted(dict_len.items(), key=operator.itemgetter(1), reverse=False)
        sorted_dict = [{item[0]: d[item [0]]} for item in sorted_key_list]
        
        if t in range(0,len(self.x_true[0]),100):
            print('----------------------------------')
            print('t: %d, x_correct: %f, highest prob: %f:' % (t, self.x_true[0, t], p.argmax(2)))
            print(sorted_dict[:5])
            print('Valor real w: ',w)
            print('palabra codificado: ',u)
            #print("x_t: ", x_t)
            print('----------------------------------')
        """
    
    message_encoded = [item for sublist in message_encoded for item in sublist]
    return message_encoded

  def Decode(self, message_encoded):
    x_t = self.x_t0
    s = None
    message_decoded = []
    word_encoded = []
    i = 0
    for t in range(len(self.x_true[0])):
        p, s = self.model.predict(x_t, s)
        w_ = p.reshape((len(p[0,0]), -1))
        w_ = w_.tolist()
        dictionary = [ (self.symbols[j], w_[j]) for j in range(len(w_)) ]
        h = make_huffman_tree(dictionary) 
        d = flatten_to_dict(h)
        
        word_encoded = []
        word_encoded.extend(message_encoded[i:i+1])
        while ( word_encoded not in d.values() ) and ( i < len(message_encoded)-1 ):
            i = i+1
            word_encoded.extend(message_encoded[i:i+1])
        i = i+1
        word_decoded = huffman_decode(word_encoded, h)
        message_decoded.append(word_decoded)
        if self.is_cuda:
            x_t = (torch.FloatTensor([[word_decoded]])).cuda()
        else:
            x_t = torch.FloatTensor([[word_decoded]])
        
    return message_decoded

  def Check(self, message_decoded):
    # ANÁLISIS DE RESULTADOS
    x_true = self.x_true[0].cpu().detach().numpy()
    message_decoded = np.array(message_decoded)
    err = abs(message_decoded-x_true)
    plt.plot(err)
    plt.title("Absolute error at each sample")
    #plt.show()
    plt.savefig('/2L_10B_16MB/absolute_sample_error.png')

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(message_decoded, color ='tab:red')
    ax.plot(x_true, color ='tab:blue')

    ax.set_title('Decoded voice representation')
    
    # display the plot
    #plt.show()
    plt.savefig('/2L_10B_16MB/decoded_voice_representation.png')

  def DataCompressionRate(self, message_encoded, output_dim):
    original_bits_sample = math.log2(output_dim)
    encoded_bits_sample = len(message_encoded)/len(self.x_true[0])
    compression_rate = original_bits_sample/encoded_bits_sample
    """
    print("Longitud del mensaje sin codificar: " + str(len(self.x_true[0])*int(original_bits_sample)))
    print("Longitud del mensaje codificado: " + str(len(message_encoded)))
    print("Original size: ", original_bits_sample, "bits/sample")
    print("Encoded size: ", encoded_bits_sample, "bits/sample")
    print("Compression rate: ", compression_rate)
    """
    return compression_rate

x_true = torch.FloatTensor( x_test )

"""plt.plot(x_true[:1,:,:1])
plt.title("Test signal shape")
#plt.show()
plt.savefig('test_shape.png')"""

x_t0 = torch.FloatTensor([[[1]]])
symbols = [i for i in range (output_dim)]
rate = 0

model.eval()
tic = time.time()
for i in range(len(x_true)):
    #print("Signal " + str(i))
    CoderDecoder = CodDecod(x_true[i:i+1], x_t0, symbols, model)
    message_encoded = CoderDecoder.Code()
    message_decoded = CoderDecoder.Decode(message_encoded)
    #CoderDecoder.Check(message_decoded)
    rate += (CoderDecoder.DataCompressionRate(message_encoded, output_dim))/len(x_true)

tac = time.time()
tiempo_codificacion = time.strftime("%H:%M:%S", time.gmtime(tac-tic))

print("Compression rate average: ", rate)
print("Tiempo de codificación: ", tiempo_codificacion, " [hh:mm:ss]")
