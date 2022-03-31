import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, Dense, ZeroPadding1D, ZeroPadding2D, Reshape, Layer, MaxPool1D, Conv2D, BatchNormalization, ReLU
from tensorflow.keras import Sequential
from tensorflow.keras.models import Model
import tensorflow_probability as tfp
from graph.sign27 import Graph
import numpy as np

class UnitGCN(Layer):
    def __init__(self, in_channels, out_channels, A, groups, num_point, coff_embedding=4, num_subset=3):
        super(UnitGCN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_point = num_point
        self.groups = groups
        self.num_subset = num_subset
        self.DecoupleA = tf.Variable(tf.reshape(A, [3,1,num_point, num_point]))
        self.DecoupleA = tf.tile(self.DecoupleA, 
                                tf.constant([1, groups, 1, 1])
                                )
        
        if in_channels != out_channels:
            self.down = lambda x: BatchNormalization()(Conv2D(out_channels,1)(x))
        else:
            self.down = lambda x: x
        
        self.add1 = tf.keras.layers.Add()
        self.add2 = tf.keras.layers.Add()
        self.bn0 = BatchNormalization()
        self.bn = BatchNormalization()
        self.relu = ReLU()
        
        self.Linear_weight = tf.Variable(
            tf.random_normal_initializer(mean=1., stddev=2.)(
            shape=[in_channels, out_channels*num_subset]),
            name='unit_gcn/linear/weight')
        self.Linear_bias = tf.Variable(
            tf.constant(1e-6, shape=[1,1,1,out_channels*num_subset]),
            name='unit_gcn/linear/bias')
        
        eye_array = []
        for i in range(out_channels):
            eye_array.append(tf.eye(num_point))
        self.eyes = tf.constant(np.array(eye_array))
	
    def norm(self, A):
        b,c,h,w = A.get_shape()
        A = tf.reshape(A, [c, self.num_point, self.num_point])
        A = tf.cast(A, tf.float32)
        D_list = tf.reduce_sum(A, axis=1)
        D_list = tf.reshape(D_list, [c,1,self.num_point])
        D_list_12 = (D_list + 0.001)**(-1)
        D_12 = tf.cast(self.eyes,tf.float32) * tf.cast(D_list_12, tf.float32)
        A = tf.linalg.matmul(A, D_12)
        A = tf.reshape(A, [b,c,h,w])
        
        return A
        
    def call(self, inputs):
        learn_A = tf.tile(self.DecoupleA, [1, self.out_channels //self.groups, 1, 1])
        norm_learn_A = tf.concat([self.norm(learn_A[0:1, ...]), self.norm(
                learn_A[1:2, ...]), self.norm(learn_A[2:3, ...])], axis=0)
        norm_learn_A = tf.transpose(norm_learn_A,[2,3,0,1])
        x = tf.einsum('ntwc,cd->ntwd', inputs, self.Linear_weight)
        x = self.add1([x, self.Linear_bias])
        x = self.bn0(x)
        
        n, t, v, kc = x.get_shape()
        x = tf.keras.layers.Reshape((t, v, self.num_subset, kc // self.num_subset))(x)
        x = tf.einsum('ntvkc,vwkc->ntwc', x, norm_learn_A)
        
        x = self.bn(x)
        x_b = self.down(inputs)
        x = self.add2([x, x_b])
        x = self.relu(x)
        return x

class DropBlockT_1d(Layer):
    def __init__(self, block_size=7):
        super(DropBlockT_1d, self).__init__()
        self.keep_prob = 0.5
        self.block_size = block_size
        self.maxpool = MaxPool1D(self.block_size, strides=1, padding='same')
        
    def call(self, inputs, keep_prob):
        self.keep_prob = keep_prob
        if self.keep_prob == 1:
            return inputs
        
        n,t,v,c = inputs.get_shape()
        
        input_abs = tf.reduce_mean(tf.reduce_mean(
                            tf.abs(inputs), axis=2),
                            axis=2)
        input_abs = tf.math.divide(input_abs, tf.reduce_sum(input_abs))*tf.cast(tf.size(input_abs), tf.float32)
        input_abs = tf.keras.layers.Reshape((t,1))(input_abs)
        
        gamma = (1. - self.keep_prob) /self.block_size
        
        input1 = tf.keras.layers.Reshape((t,c*v))(inputs)
        M = tf.clip_by_value(input_abs*gamma, 0.0, 1.0)
        p,q,r = M.get_shape()
        prob = tf.reduce_mean(M)
        
        M = tfp.distributions.Bernoulli(probs=prob).sample(sample_shape=(q,r))
        M = tf.tile(M, [1,c*v])
        M = tf.expand_dims(M, axis=0)
        print(M.get_shape())
        M = tf.cast(M, tf.float32)
        Msum = self.maxpool(M)
        mask = tf.cast((1. - Msum), tf.float32)
        
        out = tf.math.divide(input1*mask*tf.cast(tf.size(mask), tf.float32),tf.reduce_sum(mask))
        out = Reshape([t,v,c])(out)
        print(out.get_shape())
        
        return out

class DropBlock_Ske(Layer):
    def __init__(self, num_point, block_size=7):
        super(DropBlock_Ske, self).__init__()
        self.keep_prob = 0.0
        self.block_size = block_size
        self.num_point = num_point
        
    def call(self, inputs, keep_prob, A):
        self.keep_prob = keep_prob
        if self.keep_prob == 1:
            return inputs
        n,t,v,c = inputs.get_shape()
        input_abs = tf.reduce_mean(tf.reduce_mean(tf.math.abs(inputs),axis=1),axis=2)
        input_abs = tf.math.divide(input_abs, tf.reduce_sum(input_abs)) * tf.cast(tf.size(input_abs), tf.float32)
        if self.num_point == 25:  # Kinect V2
            gamma = (1. - self.keep_prob) / (1 + 1.92)
        elif self.num_point == 20:  # Kinect V1
            gamma = (1. - self.keep_prob) / (1 + 1.9)
        else:
            gamma = (1. - self.keep_prob) / (1 + 1.92)
            # warnings.warn('undefined skeleton graph')
        
        Md = tf.clip_by_value(input_abs*gamma, 0.0, 1.0)
        print(Md.get_shape())
        p,q = Md.get_shape()
        prob = tf.reduce_mean(Md)
        
        Mseed = tfp.distributions.Bernoulli(probs=prob).sample(sample_shape=(q))
        Mseed = tf.expand_dims(Mseed, axis=0)
        Mseed = tf.cast(Mseed, tf.float32)
        M = tf.matmul(Mseed, tf.expand_dims(tf.cast(A, tf.float32), axis=0))
        print(1-M)
        
        M = tf.where(tf.math.greater(M,0.001),tf.constant(1.0, tf.float32),M)
        M = tf.where(tf.math.less(M,0.5), tf.constant(0.0, tf.float32),M)
        
        mask = Reshape((1,self.num_point,1))(1.0-M)
        out = tf.math.divide(inputs * mask * tf.cast(tf.size(mask), tf.float32), tf.reduce_sum(mask))      
        return out
        
class UnitTCN(Layer):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1, num_point=27, block_size=41):
        super(UnitTCN, self).__init__()
        pad = int((kernel_size - 1) /2)
        self.pad = ZeroPadding2D(padding=(pad,0))
        self.conv = Conv2D(out_channels, kernel_size = (kernel_size,1),
                            padding = 'valid', strides=(stride,1))
        self.bn = BatchNormalization()
        self.relu = ReLU()
        
        self.dropS = DropBlock_Ske(num_point)
        self.dropT = DropBlockT_1d(block_size=block_size)
        
    def call(self, inputs, keep_prob, A):
        out = self.pad(inputs)
        out = self.conv(out)
        print(out.get_shape())
        out = self.bn(out)
        out = self.dropS(out, keep_prob, A)
        out = self.dropT(out, keep_prob)
        
        return out

class UnitTCN_skip(Layer):
    def __init__(self, out_channels, kernel_size=9, stride=1):
        super(UnitTCN_skip, self).__init__()
        pad = int((kernel_size -1)/2)
        self.pad = ZeroPadding2D(padding=(pad,0))
        self.conv = Conv2D(out_channels,kernel_size=(kernel_size,1),
                            padding='valid', strides=(stride,1))
        self.bn = BatchNormalization()
        self.relu = ReLU()
    def call(self, inputs):
        out = self.pad(inputs)
        out = self.bn(self.conv(out))
        
        return out
        
class TCN_GCN_Unit(Layer):
    def __init__(self, in_channels, out_channels,
                A, groups, num_point, block_size, 
                stride=1, residual=True, attention=True):
        super(TCN_GCN_Unit, self).__init__()
        self.gcn1 = UnitGCN(in_channels, out_channels, A, groups, num_point)
        self.tcn1 = UnitTCN(out_channels,out_channels,
                            stride=stride, num_point=num_point)
        self.relu = ReLU()
        
        self.A = tf.Variable(tf.reduce_sum(tf.reshape(tf.cast(A, tf.float32),
                                                     [3, num_point, num_point]),
                                           axis=0)
                                           )
        if not residual:
            self.residual = lambda x: tf.constant(0, tf.float32)
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        
        else:
            self.residual = UnitTCN_skip(out_channels, kernel_size=1, stride=stride)
        
        self.dropSke = DropBlock_Ske(num_point=num_point)
        self.dropT_skip = DropBlockT_1d(block_size=block_size)
        self.attention = attention
        num_jpts = A.shape[-1]
        
        if attention:
            print('Attention Enabled!')
            self.sigmoid = tf.math.sigmoid
            # Temporal Attention
            self.pad_ta = ZeroPadding1D(padding=4)
            self.conv_ta = Conv1D(1, 9, padding='valid')
            
            # Spatial Attention
            ker_jpt = num_jpts - 1 if not num_jpts % 2 else num_jpts
            pad = (ker_jpt - 1) // 2
            self.pad_sa = ZeroPadding1D(padding=pad)
            self.conv_sa = Conv1D(1, ker_jpt, padding='valid')
            
            #Channel Attention
            rr = 2
            self.fc1c = Dense(out_channels // rr)
            self.fc2c = Dense(out_channels)
    
    def call(self, inputs, keep_prob):
        out = self.gcn1(inputs)
        if self.attention:
            # spatial attention
            se = tf.reduce_mean(out, axis=1)  # N C V
            se = self.pad_sa(se)
            se1 = self.sigmoid(self.conv_sa(se))
            out = out * tf.expand_dims(se1,axis=1) + out
            # a1 = se1.unsqueeze(-2)

            # temporal attention
            se = tf.reduce_mean(out, axis=2)
            se = self.pad_ta(se)
            se1 = self.sigmoid(self.conv_ta(se))
            out = out * tf.expand_dims(se1,axis=2) + out
            # a2 = se1.unsqueeze(-1)

            # channel attention
            se = tf.reduce_mean(out, axis=[1,2])
            se1 = self.relu(self.fc1c(se))
            se2 = self.sigmoid(self.fc2c(se1))
            out = out * tf.expand_dims(tf.expand_dims(se2, axis=1),axis=1) + out
            # a3 = se2.unsqueeze(-1).unsqueeze(-1)
        
        out = self.tcn1(out, keep_prob, self.A)
        x_skip = self.dropSke(self.residual(inputs), keep_prob, self.A)
        x_skip = self.dropT_skip(x_skip, keep_prob)
        
        return self.relu(out + x_skip)

class SLGCN(Model):
    def __init__(self, num_class=60, num_point=27, num_person=1,
                groups=8, block_size=41, in_channels=3):
        super(SLGCN, self).__init__()
        A = Graph('spatial').A
        self.data_bn = BatchNormalization()
        self.l1 = TCN_GCN_Unit(in_channels, 64, A, groups, num_point, block_size,
                               residual=False)
        self.l2 = TCN_GCN_Unit(64, 64, A, groups, num_point, block_size)
        self.l3 = TCN_GCN_Unit(64, 64, A, groups, num_point, block_size)
        self.l4 = TCN_GCN_Unit(64, 64, A, groups, num_point, block_size)
        self.l5 = TCN_GCN_Unit(64, 128, A, groups, num_point, block_size, stride=2)
        self.l6 = TCN_GCN_Unit(128, 128, A, groups, num_point, block_size)
        self.l7 = TCN_GCN_Unit(128, 128, A, groups, num_point, block_size)
        self.l8 = TCN_GCN_Unit(128, 256, A, groups, num_point, block_size, stride=2)
        self.l9 = TCN_GCN_Unit(256, 256, A, groups, num_point, block_size)
        self.l10 = TCN_GCN_Unit(256, 256, A, groups, num_point, block_size)
        
        self.fc = Dense(num_class)
        
    def call(self, inputs, keep_prob=0.8):
        N,T,V,M,C = inputs.get_shape()
        x = Reshape([T, V*M*C])(inputs)
        x = self.data_bn(x)
        x = Reshape([T,V,M,C])(x)
        x = tf.transpose(x,[0,3,1,2,4])
        sh = tf.shape(x)
        x = tf.reshape(x, [sh[0]*sh[1], sh[2], sh[3], sh[4]])
        
        x = self.l1(x, 1.0)
        x = self.l2(x, 1.0)
        x = self.l3(x, 1.0)
        x = self.l4(x, 1.0)
        x = self.l5(x, 1.0)
        x = self.l6(x, 1.0)
        x = self.l7(x, keep_prob)
        x = self.l8(x, keep_prob)
        x = self.l9(x, keep_prob)
        x = self.l10(x, keep_prob)
        
        sh_new = tf.shape(x)
        
        x = tf.reshape(x, [sh[0],sh[1],sh_new[1]*sh_new[2],sh_new[3]])
        x = tf.reduce_mean(x, axis=2)
        x = tf.reduce_mean(x, axis=1)
        
        
        return self.fc(x)

A = Graph('spatial').A
model = SLGCN()  
inp = Input([8,27,2,3])
# inp = tf.ones([1,8,27,64])
out = model(inp) 
m = Model(inp, out)

for x in m.trainable_variables:
    print(x.name)
