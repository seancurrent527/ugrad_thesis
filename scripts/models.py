from keras.layers import Dense, Input, Concatenate, Dropout, Lambda
from keras.regularizers import l2, l1, l1_l2
from keras.models import Model
import keras.backend as K

def stateless_model(input_shape, wrap = 3):
    x0 = Input(input_shape)

    def inner_model(input_shape):
        xin = Input(input_shape)
        xhid1 = Dense(32, activation='relu', kernel_regularizer=l1(0.15))(xin)
        xhid1 = Dropout(0.5)(xhid1)
        xhid2 = Concatenate()([xin, xhid1])
        
        xout = Dense(input_shape[0])(xhid2)

        xhid3 = Concatenate()([xout, xhid2])

        combiner = Dense(input_shape[0], activation = 'sigmoid')(xhid3)
        xout = Lambda(lambda x: x[1] * x[0] + (1 - x[0]) * x[2])([combiner, xin, xout])

        return Model(xin, xout)

    x_ = x0
    outs = []
    inner = inner_model(input_shape)
    for _ in range(wrap):
        x_ = inner(x_)
        outs.append(x_)

    if len(outs) > 1:
        xout = Concatenate()(outs)
    else:
        xout = outs[0]

    return Model(x0, xout), inner

def state_model(input_shape, wrap = 3):
    x0 = Input(input_shape)
    s0 =  Lambda(lambda x: K.zeros_like(x))(x0)

    def inner_model(input_shape):
        x, s = Input(input_shape, name = 'Xt'), Input(input_shape)
        h = Concatenate()([x, s])
        
        reset = Dense(input_shape[-1], activation = 'sigmoid')(h)
        hs = Lambda(lambda x: x[0] * x[1])([s, reset])
        hr = Concatenate()([hs, x])

        update = Dense(input_shape[-1], activation = 'sigmoid')(h)

        hhat = Dense(input_shape[-1], activation = 'tanh')(hr)
        
        ns = Lambda(lambda x: (1 - x[0]) * x[1] + (x[0] * x[2]))([update, s, hhat])

        xout = Dense(input_shape[-1], activation = 'relu')(ns)

        return Model([x, s], [xout, ns])
    
    x_, s_ = x0, s0
    outs = []
    inner = inner_model(input_shape)
    for _ in range(wrap):
        x_, s_ = inner([x_, s_])
        outs.append(x_)

    if len(outs) > 1:
        xout = Concatenate()(outs)
    else:
        xout = outs[0]

    return Model(x0, xout), inner

def get_model(fname, input_shape, wrap = 3, load = True, state = False):
    if not state:
        model, inner = stateless_model(input_shape, wrap=wrap)
    else:
        model, inner = state_model(input_shape, wrap=wrap)
    if load:
        model.load_weights(fname)
    return model, inner