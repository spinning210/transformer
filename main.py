from __future__ import print_function
from keras.preprocessing import sequence
from keras.datasets import imdb
from keras import backend as K
from keras.engine.topology import Layer
from keras.models import Model
from keras.layers import *
import tensorflow as tf

from loguru import logger


class Position_Embedding(Layer):
    
    def __init__(self, size=None, mode='sum', **kwargs):
        super(Position_Embedding, self).__init__(**kwargs)    
        self.size = size
        self.mode = mode
        
        
    def call(self, x):
        if (self.size == None) or (self.mode == 'sum'):
            self.size = int(x.shape[-1])
        batch_size,seq_len = K.shape(x)[0],K.shape(x)[1]
        position_j = 1.0 / K.pow(10000.0,2 * K.arange(self.size / 2, dtype='float32') / self.size)
        #self.my_print(position_j)
        position_j = K.expand_dims(position_j, 0) #在索引position_j的 axis=0 轴，添加 1 个尺寸的维度。
        position_i = K.cumsum(K.ones_like(x[:,:,0]), 1)-1 #K.arange不支持变长，只好用这种方法生成
        position_i = K.expand_dims(position_i, 2)
        position_ij = K.dot(position_i, position_j)
        position_ij = K.concatenate([K.cos(position_ij), K.sin(position_ij)], 2)
        
        if self.mode == 'sum':
            return position_ij + x
        elif self.mode == 'concat':
            self.my_print(K.concatenate([position_ij, x], 2))
            return K.concatenate([position_ij, x], 2)

    def my_print(self,x):
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            print(x.eval())
        
    def compute_output_shape(self, input_shape):
        if self.mode == 'sum':
            return input_shape
        elif self.mode == 'concat':
            return (input_shape[0], input_shape[1], input_shape[2]+self.size)


class Attention(Layer):

    def __init__(self, nb_head, size_per_head, mask_right=False, **kwargs):
        self.nb_head = nb_head
        self.size_per_head = size_per_head
        self.output_dim = nb_head*size_per_head
        self.mask_right = mask_right
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.WQ = self.add_weight(name='WQ', 
                                  shape=(input_shape[0][-1], self.output_dim),
                                  initializer='glorot_uniform', #均匀分布初始化
                                  trainable=True)
        self.WK = self.add_weight(name='WK', 
                                  shape=(input_shape[1][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.WV = self.add_weight(name='WV', 
                                  shape=(input_shape[2][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        super(Attention, self).build(input_shape)
        
    def Mask(self, inputs, seq_len, mode='mul'):
        if seq_len == None:
            return inputs
        else:
            mask = K.one_hot(seq_len[:,0], K.shape(inputs)[1])
            mask = 1 - K.cumsum(mask, 1)
            for _ in range(len(inputs.shape)-2):
                mask = K.expand_dims(mask, 2)
            if mode == 'mul':
                return inputs * mask
            if mode == 'add':
                return inputs - (1 - mask) * 1e12 
                
    def call(self, x):
        #如果只傳入Q_seq,K_seq,V_seq，不做Mask
        #如果同時傳入Q_seq,K_seq,V_seq,Q_len,V_len，多餘部分做Mask
        if len(x) == 3:
            Q_seq,K_seq,V_seq = x
            Q_len,V_len = None,None
        elif len(x) == 5:
            Q_seq,K_seq,V_seq,Q_len,V_len = x

        #==Q K V Linear
        Q_seq = K.dot(Q_seq, self.WQ) #(,80,128)
        Q_seq = K.reshape(Q_seq, (-1, K.shape(Q_seq)[1], self.nb_head, self.size_per_head)) #(,80,,16)*8
        Q_seq = K.permute_dimensions(Q_seq, (0,2,1,3)) #transpose  (,8,80,16)

        K_seq = K.dot(K_seq, self.WK)
        K_seq = K.reshape(K_seq, (-1, K.shape(K_seq)[1], self.nb_head, self.size_per_head))
        K_seq = K.permute_dimensions(K_seq, (0,2,1,3))

        V_seq = K.dot(V_seq, self.WV)
        V_seq = K.reshape(V_seq, (-1, K.shape(V_seq)[1], self.nb_head, self.size_per_head))
        V_seq = K.permute_dimensions(V_seq, (0,2,1,3))

        #==softmax(Q*Kt/dk**0.5)*V
        A = K.batch_dot(Q_seq, K_seq, axes=[3,3]) / self.size_per_head**0.5
        A = K.permute_dimensions(A, (0,3,2,1))
        #A = self.Mask(A, V_len, 'add')
        #A = K.permute_dimensions(A, (0,3,2,1)) 
        # if self.mask_right:
        #     ones = K.ones_like(A[:1, :1])
        #     mask = (ones - K.tf.matrix_band_part(ones, -1, 0)) * 1e12
        #     A = A - mask
        A = K.softmax(A)
        O_seq = K.batch_dot(A, V_seq, axes=[3,2])
        O_seq = K.permute_dimensions(O_seq, (0,2,1,3))
        O_seq = K.reshape(O_seq, (-1, K.shape(O_seq)[1], self.output_dim))
        #O_seq = self.Mask(O_seq, Q_len, 'mul')
        return O_seq
        
    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], self.output_dim)

max_features = 20000 #出現沒超過20000次
maxlen = 80
batch_size = 32

logger.info('Loading data...')
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words = max_features) #already word2num
logger.info(str(len(x_train)) + ' train sequences')
logger.info(str(len(x_test)) + ' test sequences')

logger.info('Pad sequences (samples x time)...')
x_train = sequence.pad_sequences(x_train, maxlen = maxlen)
x_test = sequence.pad_sequences(x_test, maxlen = maxlen)
logger.info('x_train shape:' + str(x_train.shape))
logger.info('x_test shape:' + str(x_test.shape))



S_inputs = Input(shape=(None,), dtype='int32')
embeddings = Embedding(max_features, 128)(S_inputs)  #max_features = 字典长度 ||  128 = 全连接嵌入的维度
embeddings = Position_Embedding()(embeddings) 
O_seq = Attention(8,16)([embeddings,embeddings,embeddings])
O_seq = GlobalAveragePooling1D()(O_seq)
O_seq = Dropout(0.5)(O_seq)
outputs = Dense(1, activation='sigmoid')(O_seq)

model = Model(inputs=S_inputs, outputs=outputs)

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

logger.info('Training...')
model.fit(x_train, y_train,batch_size=batch_size,epochs=10,validation_data=(x_test, y_test))
