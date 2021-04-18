import tensorflow as tf

from keras.models import model_from_json
from keras.models import load_model

from keras.models import Model
from keras.layers import Input, LSTM, Dense, Add, Softmax,Reshape
from keras.layers import Lambda,Concatenate,Flatten,ConvLSTM2D
from keras.layers import Permute,Conv2D
from keras import backend as K
from keras.layers import BatchNormalization
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau,EarlyStopping
# from tensorflow.keras.metrics import KLDivergence
import keras.losses
import os

import keras_config as cfg
import pickle as pk
import numpy as np
# import data_generator_heat_map as dg
# from utilities import *

class DecayLayer(keras.layers.Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(DecayLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        print("in build: ", input_shape)
        # Create a trainable weight variable for this layer.
        self._A = self.add_weight(name='A', 
                                    shape=(input_shape[1], self.output_dim),
                                    initializer='uniform',
                                    trainable=True)
        self._B = self.add_weight(name='B', 
                                    shape=(input_shape[1], self.output_dim),
                                    initializer='uniform',
                                    trainable=True)
        super(DecayLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        # print("x is: ", x)
        return tf.math.divide(1, tf.math.exp(self._A*x + self._B))

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

def mul_sca(x):
    return x[0]*x[1]


def load_decay_model():
    # arch_path = './arch/non_heatmap/model_architecture.json'
    model_path = './keras_models/heatmap_decay/new.h5'
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    Concatenatelayer1 = Concatenate(axis=-1)

    expand_dim_layer = Lambda(lambda x: K.expand_dims(x,1))
    expand_dim_layer2 = Lambda(lambda x: K.expand_dims(x,2))

    # expand_dim_layer1 = Lambda(lambda x: K.expand_dims(x,axis=1))
    # get_dim_layer = Lambda(lambda x: x[:,0,0,:,:])
    get_dim_layer1 = Lambda(lambda x: x[:,0,:,:,:])
    # flatten_layer = Flatten()

    # scale_layer = Lambda(lambda x:x/K.sum(x,axis=(1,2)))
    # scale_layer1 = Lambda(lambda x:x/K.sum(x,axis=(1)))
    scale_layer2 = Lambda(lambda x:x/K.sum(x,axis=(1,2,3)))

    temporal_decay = DecayLayer(1)
    num_decay = DecayLayer(1)
    # configuration
    kernel_size = cfg.conv_kernel_size
    latent_dim = cfg.latent_dim
    row = cfg.num_row
    col = cfg.num_col
    epochs = 1000

    input_shape1 = (cfg.running_length,row,col,1)           # Sample, time, row, col, channel
    input_shape2 = (cfg.running_length,row,col,latent_dim*2)
    input_shape3 = (cfg.running_length,row,col,latent_dim)


    # convLSTM for target past segment average
    encoder_inputs = Input(shape=(cfg.running_length, row, col, 1))
    convlstm_encoder = ConvLSTM2D(filters=latent_dim*2, kernel_size=(kernel_size, kernel_size),
                       input_shape=input_shape1, dropout=cfg.dropout_rate, recurrent_dropout=0.1,
                       stateful=cfg.stateful_across_batch, data_format = 'channels_last',
                       padding='same', return_sequences=True, return_state=True)
    pst_outputs_sqns, pst_state_h0, pst_state_c0 = convlstm_encoder(encoder_inputs)
    states0 = [pst_state_h0, pst_state_c0]
    # print(convlstm_encoder)

    convlstm_encoder1 = ConvLSTM2D(filters=latent_dim, kernel_size=(kernel_size, kernel_size),
                       input_shape=input_shape2, dropout=cfg.dropout_rate, recurrent_dropout=0.1,
                       stateful=cfg.stateful_across_batch, data_format = 'channels_last',
                       padding='same', return_sequences=True, return_state=True)
    pst_outputs_sqns, pst_state_h1, pst_state_c1 = convlstm_encoder1(pst_outputs_sqns)
    states1 = [pst_state_h1, pst_state_c1]

    convlstm_encoder2 = ConvLSTM2D(filters=latent_dim//2, kernel_size=(kernel_size, kernel_size),
                       input_shape=input_shape3, dropout=cfg.dropout_rate, recurrent_dropout=0.1,
                        stateful=cfg.stateful_across_batch, data_format = 'channels_last',
                        padding='same', return_sequences=True, return_state=True)
    # print(pst_outputs_sqns.shape())
    pst_outputs_sqns, pst_state_h2, pst_state_c2 = convlstm_encoder2(pst_outputs_sqns)
    states2 = [pst_state_h2, pst_state_c2]

    # print(pst_outputs_sqns)

    dinput_shape1 = (1,row,col,1)           # Sample, time, row, col, channel
    dinput_shape2 = (1,row,col,latent_dim*2)
    dinput_shape3 = (1,row,col,latent_dim)
    # ###======convLSTM on target future decoder======
    decoder_inputs = Input(shape=(1,row,col,1))   # Only last sequence from encoder
    convlstm_decoder = ConvLSTM2D(filters=latent_dim*2, kernel_size=(kernel_size, kernel_size),
                       input_shape=dinput_shape1,dropout=cfg.dropout_rate, recurrent_dropout=0.1,
                       stateful=cfg.stateful_across_batch, data_format = 'channels_last',
                       padding='same', return_sequences=True, return_state=True)

    convlstm_decoder1 = ConvLSTM2D(filters=latent_dim, kernel_size=(kernel_size, kernel_size),
                       input_shape=dinput_shape2,dropout=cfg.dropout_rate, recurrent_dropout=0.1,
                       stateful=cfg.stateful_across_batch, data_format = 'channels_last',
                       padding='same', return_sequences=True, return_state=True)

    convlstm_decoder2 = ConvLSTM2D(filters=latent_dim//2, kernel_size=(kernel_size, kernel_size),
                       input_shape=dinput_shape3,dropout=cfg.dropout_rate, recurrent_dropout=0.1,
                       stateful=cfg.stateful_across_batch, data_format = 'channels_last',
                       padding='same', return_sequences=True, return_state=True)


    # ### 2D conv
    pred_conv_conv = Conv2D(filters=128, kernel_size=(kernel_size,kernel_size), padding='same',
        activation='relu', use_bias=True, kernel_initializer='glorot_uniform')
    pred_conv_conv1 = Conv2D(filters=256, kernel_size=(kernel_size,kernel_size), padding='same',
        activation='relu', use_bias=True, kernel_initializer='glorot_uniform')
    pred_conv_conv2 = Conv2D(filters=1, kernel_size=(kernel_size,kernel_size), padding='same',   # filters=fps
        activation='relu', use_bias=True, kernel_initializer='glorot_uniform')

    # 2D conv for other users' gt
    others_conv0 = Conv2D(filters=4, kernel_size=(kernel_size,kernel_size), padding='same',
        activation='relu', use_bias=True, kernel_initializer='glorot_uniform')
    others_conv1 = Conv2D(filters=8, kernel_size=(kernel_size,kernel_size), padding='same',
        activation='relu', use_bias=True, kernel_initializer='glorot_uniform')
    # others_conv2 = Conv2D(filters=1, kernel_size=(kernel_size,kernel_size), padding='same',   # filters=fps
    #     activation='relu', use_bias=True, kernel_initializer='glorot_uniform')

    # 2D conv for other users' var gt
    others_var_conv0 = Conv2D(filters=2, kernel_size=(kernel_size,kernel_size), padding='same',
        activation='relu', use_bias=True, kernel_initializer='glorot_uniform')
    others_var_conv1 = Conv2D(filters=4, kernel_size=(kernel_size,kernel_size), padding='same',
        activation='relu', use_bias=True, kernel_initializer='glorot_uniform')
    # others_var_conv2 = Conv2D(filters=1, kernel_size=(kernel_size,kernel_size), padding='same',   # filters=fps
    #     activation='relu', use_bias=True, kernel_initializer='glorot_uniform')

    # num_conv1 = Conv2D(filters=4, kernel_size=(kernel_size,kernel_size), padding='same',
    #     activation='relu', use_bias=True, kernel_initializer='glorot_uniform')

    # pred_interval_conv1 = Conv2D(filters=4, kernel_size=(kernel_size,kernel_size), padding='same',
    #     activation='relu', use_bias=True, kernel_initializer='glorot_uniform')

    # Dense for number of users
    # other_user_dense = Dense(8, input_shape=(1,),activation='relu')

    # output final map for next timestamp
    # final_output = Dense(row*col, input_shape=(2*row*col+16,),activation='softmax')
    # final_reshape = Reshape((row,col,1), input_shape=(row*col,))

    # bnlayer0 = BatchNormalization(axis=-1,center=True,scale=True)
    # bnlayer1 = BatchNormalization(axis=-1,center=True,scale=True)
    # bnlayer2 = BatchNormalization(axis=-1,center=True,scale=True)

    # 
    all_outputs= []
    inputs = decoder_inputs
    other_inputs = Input(shape=(cfg.predict_step, row, col, 1))
    other_inputs_var = Input(shape=(cfg.predict_step, row, col, 1))
    num_other = Input(shape=(cfg.predict_step,1))
    pred_inteval = Input(shape=(cfg.predict_step,1))

    # num_users = Input(shape=(1))
    for time_ind in range(cfg.predict_step):
        # print('k0', inputs.shape)
        fut_outputs_sqns0, fut_state_h, fut_state_c = convlstm_decoder([inputs]+states0)
        states0 = [fut_state_h, fut_state_c]
        fut_outputs_sqns1, fut_state_h, fut_state_c = convlstm_decoder1([fut_outputs_sqns0]+states1)
        states1 = [fut_state_h, fut_state_c]
        fut_outputs_sqns2, fut_state_h, fut_state_c = convlstm_decoder2([fut_outputs_sqns1]+states2)
        states2 = [fut_state_h, fut_state_c]

        fut_outputs_sqns = Concatenatelayer1([fut_outputs_sqns0,fut_outputs_sqns1,fut_outputs_sqns2])
        print('decoder output shape: ', fut_outputs_sqns.shape)
        # Concatenate ground truth and var

        groud_truth_map = other_inputs[:,time_ind,:,:,:]
        # print(groud_truth_map.shape)
        groud_truth_map_var = other_inputs_var[:,time_ind,:,:,:]


        num_other_gt = num_other[:,time_ind]
        print("num other gt shape ", num_other_gt.shape)
        pred_interval_gt = pred_inteval[:,time_ind]
        print('ground truth shape lllllllllll', groud_truth_map.shape)
        groud_truth_map = others_conv0(groud_truth_map)
        groud_truth_map = others_conv1(groud_truth_map)

        groud_truth_map_var = others_var_conv0(groud_truth_map_var)
        groud_truth_map_var = others_var_conv1(groud_truth_map_var)    
        print('ground truth shape', groud_truth_map_var.shape)
        # print(groud_truth_map_var.shape)
        groud_truth_map = expand_dim_layer(groud_truth_map)
        groud_truth_map_var = expand_dim_layer(groud_truth_map_var)
        others_info = Concatenatelayer1([groud_truth_map, groud_truth_map_var])
        print("concatenated other shape: ", others_info.shape)

        ## Not treated as feature map anymore
        # num_other_map = num_conv1(num_other_gt)
        # print('num map shape', num_other_map.shape)
        # pre_interval_map = pred_interval_conv1(pred_interval_gt)
        # print('l', pre_interval_map.shape)


        temporal_value = temporal_decay(pred_interval_gt)
        temporal_value = expand_dim_layer(temporal_value)
        temporal_value = expand_dim_layer(temporal_value)
        temporal_value = expand_dim_layer(temporal_value)
        temporal_value = K.repeat_elements(temporal_value, rep=56, axis=4)
        temporal_value = K.repeat_elements(temporal_value, rep=cfg.num_col, axis=3)
        temporal_value = K.repeat_elements(temporal_value, rep=cfg.num_row, axis=2)


        num_value = num_decay(num_other_gt)
        num_value = expand_dim_layer(num_value)
        num_value = expand_dim_layer(num_value)
        num_value = expand_dim_layer(num_value)
        num_value = K.repeat_elements(num_value, rep=12, axis=4)
        num_value = K.repeat_elements(num_value, rep=cfg.num_col, axis=3)
        num_value = K.repeat_elements(num_value, rep=cfg.num_row, axis=2)

        # Multiply
        print("num value shape: ", num_value.shape)
        print("temporal shape: ", temporal_value.shape)
        fut_outputs_sqns = Lambda(mul_sca)([fut_outputs_sqns,temporal_value])
        others_info = Lambda(mul_sca)([others_info,num_value])

        # num_other_map = expand_dim_layer(num_other_map)
        # pre_interval_map = expand_dim_layer(pre_interval_map)

        fut_outputs_sqns = Concatenatelayer1([fut_outputs_sqns, others_info])
        print("final output shape: ", fut_outputs_sqns.shape)
        # fut_outputs_sqns = bnlayer(fut_outputs_sqns)
        ### predict others' future
        outputs = get_dim_layer1(fut_outputs_sqns)
        # print('k1', outputs.shape)
        outputs = pred_conv_conv(outputs)
        # print('k1', outputs.shape)
        # outputs = bnlayer0(outputs)
        outputs = pred_conv_conv1(outputs)
        # outputs = bnlayer1(outputs)
        outputs = pred_conv_conv2(outputs)

        print('f3', outputs.shape)
        outputs = expand_dim_layer(outputs)
        print('final', outputs.shape)
        inputs = outputs

        outputs = scale_layer2(outputs)
        print('sfae', outputs.shape)
        all_outputs.append(outputs)

    # Concatenate all predictions
    # print('all_outputs', all_outputs)
    decoder_outputs = Lambda(lambda x: K.concatenate(x, axis=1))(all_outputs)
    final_shape = Reshape((cfg.predict_step,row*col), input_shape=(cfg.predict_step,row,col,1))
    decoder_outputs = final_shape(decoder_outputs)




    print('encoder_input', encoder_inputs.shape)
    print('decoder_input:', decoder_inputs.shape)
    print('other_inputs: ', other_inputs.shape)
    print('other_inputs_var: ', other_inputs_var.shape)

    model = Model([encoder_inputs,decoder_inputs,other_inputs,other_inputs_var,num_other,pred_inteval],decoder_outputs)
    model.load_weights(model_path)
    return model

def load_keras_target_model():
    arch_path = './arch/non_heatmap/model_architecture.json'
    model_path = './keras_models/non_heatmap/new.h5'

    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    # import tensorflow as tf
    # print("Num GPUs Available: ", len(tensorflow.config.experimental.list_physical_devices('GPU')))

    # Utilities
    # Concatenatelayer = Concatenate(axis=2)
    Concatenatelayer1 = Concatenate(axis=-1)
    expand_dim_layer = Lambda(lambda x: K.expand_dims(x,1))
    # expand_dim_layer1 = Lambda(lambda x: K.expand_dims(x,axis=1))
    # get_dim_layer = Lambda(lambda x: x[:,0,0,:,:])
    get_dim_layer1 = Lambda(lambda x: x[:,0,:,:,:])
    # flatten_layer = Flatten()

    # scale_layer = Lambda(lambda x:x/K.sum(x,axis=(1,2)))
    # scale_layer1 = Lambda(lambda x:x/K.sum(x,axis=(1)))
    scale_layer2 = Lambda(lambda x:x/K.sum(x,axis=(1,2,3)))
    # configuration
    kernel_size = cfg.conv_kernel_size
    latent_dim = cfg.latent_dim
    row = cfg.num_row
    col = cfg.num_col
    epochs = 1000

    input_shape1 = (cfg.running_length,row,col,1)           # Sample, time, row, col, channel
    input_shape2 = (cfg.running_length,row,col,latent_dim*2)
    input_shape3 = (cfg.running_length,row,col,latent_dim)


    # convLSTM for target past segment average
    encoder_inputs = Input(shape=(cfg.running_length, row, col, 1))
    convlstm_encoder = ConvLSTM2D(filters=latent_dim*2, kernel_size=(kernel_size, kernel_size),
                       input_shape=input_shape1, dropout=cfg.dropout_rate, recurrent_dropout=0.0,
                       stateful=cfg.stateful_across_batch, data_format = 'channels_last',
                       padding='same', return_sequences=True, return_state=True)
    pst_outputs_sqns, pst_state_h0, pst_state_c0 = convlstm_encoder(encoder_inputs)
    states0 = [pst_state_h0, pst_state_c0]
    # print(convlstm_encoder)

    convlstm_encoder1 = ConvLSTM2D(filters=latent_dim, kernel_size=(kernel_size, kernel_size),
                       input_shape=input_shape2, dropout=cfg.dropout_rate, recurrent_dropout=0.0,
                       stateful=cfg.stateful_across_batch, data_format = 'channels_last',
                       padding='same', return_sequences=True, return_state=True)
    pst_outputs_sqns, pst_state_h1, pst_state_c1 = convlstm_encoder1(pst_outputs_sqns)
    states1 = [pst_state_h1, pst_state_c1]

    convlstm_encoder2 = ConvLSTM2D(filters=latent_dim//2, kernel_size=(kernel_size, kernel_size),
                       input_shape=input_shape3, dropout=cfg.dropout_rate, recurrent_dropout=0.0,
                        stateful=cfg.stateful_across_batch, data_format = 'channels_last',
                        padding='same', return_sequences=True, return_state=True)
    # print(pst_outputs_sqns.shape())
    pst_outputs_sqns, pst_state_h2, pst_state_c2 = convlstm_encoder2(pst_outputs_sqns)
    states2 = [pst_state_h2, pst_state_c2]

    # print(pst_outputs_sqns)

    # ###======convLSTM on target future decoder======
    decoder_inputs = Input(shape=(1,row,col,1))   # Only last sequence from encoder
    convlstm_decoder = ConvLSTM2D(filters=latent_dim*2, kernel_size=(kernel_size, kernel_size),
                       input_shape=input_shape1,dropout=cfg.dropout_rate, recurrent_dropout=0.0,
                       stateful=cfg.stateful_across_batch, data_format = 'channels_last',
                       padding='same', return_sequences=True, return_state=True)

    convlstm_decoder1 = ConvLSTM2D(filters=latent_dim, kernel_size=(kernel_size, kernel_size),
                       input_shape=input_shape2,dropout=cfg.dropout_rate, recurrent_dropout=0.0,
                       stateful=cfg.stateful_across_batch, data_format = 'channels_last',
                       padding='same', return_sequences=True, return_state=True)

    convlstm_decoder2 = ConvLSTM2D(filters=latent_dim//2, kernel_size=(kernel_size, kernel_size),
                       input_shape=input_shape3,dropout=cfg.dropout_rate, recurrent_dropout=0.0,
                       stateful=cfg.stateful_across_batch, data_format = 'channels_last',
                       padding='same', return_sequences=True, return_state=True)


    # ### 2D conv
    pred_conv_conv = Conv2D(filters=16, kernel_size=(kernel_size,kernel_size), padding='same',
        activation='relu', use_bias=True, kernel_initializer='glorot_uniform')
    pred_conv_conv1 = Conv2D(filters=32, kernel_size=(kernel_size,kernel_size), padding='same',
        activation='relu', use_bias=True, kernel_initializer='glorot_uniform')
    pred_conv_conv2 = Conv2D(filters=1, kernel_size=(kernel_size,kernel_size), padding='same',   # filters=fps
        activation='relu', use_bias=True, kernel_initializer='glorot_uniform')


    pred_interval_conv1 = Conv2D(filters=4, kernel_size=(kernel_size,kernel_size), padding='same',
        activation='relu', use_bias=True, kernel_initializer='glorot_uniform')

    # 2D conv for other users' gt
    # others_conv0 = Conv2D(filters=16, kernel_size=(kernel_size,kernel_size), padding='same',
    #     activation='relu', use_bias=True, kernel_initializer='glorot_uniform')
    # others_conv1 = Conv2D(filters=32, kernel_size=(kernel_size,kernel_size), padding='same',
    #     activation='relu', use_bias=True, kernel_initializer='glorot_uniform')
    # # others_conv2 = Conv2D(filters=1, kernel_size=(kernel_size,kernel_size), padding='same',   # filters=fps
    # #     activation='relu', use_bias=True, kernel_initializer='glorot_uniform')

    # # 2D conv for other users' var gt
    # others_var_conv0 = Conv2D(filters=16, kernel_size=(kernel_size,kernel_size), padding='same',
    #     activation='relu', use_bias=True, kernel_initializer='glorot_uniform')
    # others_var_conv1 = Conv2D(filters=32, kernel_size=(kernel_size,kernel_size), padding='same',
    #     activation='relu', use_bias=True, kernel_initializer='glorot_uniform')
    # others_var_conv2 = Conv2D(filters=1, kernel_size=(kernel_size,kernel_size), padding='same',   # filters=fps
    #     activation='relu', use_bias=True, kernel_initializer='glorot_uniform')

    # Dense for number of users
    # other_user_dense = Dense(8, input_shape=(1,),activation='relu')

    # output final map for next timestamp
    # final_output = Dense(row*col, input_shape=(2*row*col+16,),activation='softmax')
    # final_reshape = Reshape((row,col,1), input_shape=(row*col,))

    # bnlayer0 = BatchNormalization(axis=-1,center=True,scale=True)
    # bnlayer1 = BatchNormalization(axis=-1,center=True,scale=True)
    # bnlayer2 = BatchNormalization(axis=-1,center=True,scale=True)

    # 
    all_outputs= []
    inputs = decoder_inputs
    # other_inputs = Input(shape=(cfg.predict_step, row, col, 1))
    # other_inputs_var = Input(shape=(cfg.predict_step, row, col, 1))
    pred_inteval = Input(shape=(cfg.predict_step, row, col, 1))

    # num_users = Input(shape=(1))
    for time_ind in range(cfg.predict_step):
        # print('k0', inputs.shape)
        fut_outputs_sqns0, fut_state_h, fut_state_c = convlstm_decoder([inputs]+states0)
        states0 = [fut_state_h, fut_state_c]
        fut_outputs_sqns1, fut_state_h, fut_state_c = convlstm_decoder1([fut_outputs_sqns0]+states1)
        states1 = [fut_state_h, fut_state_c]
        fut_outputs_sqns2, fut_state_h, fut_state_c = convlstm_decoder2([fut_outputs_sqns1]+states2)
        states2 = [fut_state_h, fut_state_c]

        fut_outputs_sqns = Concatenatelayer1([fut_outputs_sqns0,fut_outputs_sqns1,fut_outputs_sqns2])
        # print(fut_outputs_sqns.shape)
        
        ## For coolborative prediction
        # Concatenate ground truth and var
        # groud_truth_map = other_inputs[:,time_ind,:,:,:]
        # # print(groud_truth_map.shape)
        # groud_truth_map_var = other_inputs_var[:,time_ind,:,:,:]

        # groud_truth_map = others_conv0(groud_truth_map)
        # groud_truth_map = others_conv1(groud_truth_map)
        pred_interval_gt = pred_inteval[:,time_ind,:,:,:]

        pre_interval_map = pred_interval_conv1(pred_interval_gt)
        pre_interval_map = expand_dim_layer(pre_interval_map)

        # groud_truth_map_var = others_var_conv0(groud_truth_map_var)
        # groud_truth_map_var = others_var_conv1(groud_truth_map_var)    

        # # print(groud_truth_map_var.shape)

        # groud_truth_map = expand_dim_layer(groud_truth_map)
        # groud_truth_map_var = expand_dim_layer(groud_truth_map_var)

        # fut_outputs_sqns = Concatenatelayer1([fut_outputs_sqns, groud_truth_map, groud_truth_map_var])

        fut_outputs_sqns = Concatenatelayer1([fut_outputs_sqns, pre_interval_map])

        ### predict others' future
        outputs = get_dim_layer1(fut_outputs_sqns)
        # print('k1', outputs.shape)
        outputs = pred_conv_conv(outputs)
        # print('k1', outputs.shape)
        # outputs = bnlayer0(outputs)
        outputs = pred_conv_conv1(outputs)
        # outputs = bnlayer1(outputs)
        outputs = pred_conv_conv2(outputs)

        # print('f3', outputs.shape)
        outputs = scale_layer2(outputs)
        outputs = expand_dim_layer(outputs)
        # print('final', outputs.shape)
        inputs = outputs

        all_outputs.append(outputs)

    # Concatenate all predictions
    # print('all_outputs', all_outputs)
    decoder_outputs = Lambda(lambda x: K.concatenate(x, axis=1))(all_outputs)
    final_shape = Reshape((cfg.predict_step,row*col), input_shape=(cfg.predict_step,row,col,1))
    decoder_outputs = final_shape(decoder_outputs)

    print('encoder_input', encoder_inputs.shape)
    print('decoder_input:', decoder_inputs.shape)
    # print('other_inputs: ', other_inputs.shape)
    # print('other_inputs_var: ', other_inputs_var.shape)

    model = Model([encoder_inputs,decoder_inputs, pred_inteval],decoder_outputs)
    model.load_weights(model_path)
    return model

def load_keras_model(heatmap = True):
    arch_path = './arch/heatmap/model_architecture.json'
    model_path = './keras_models/heatmap/best_new.h5'
    

    # with open(arch_path, 'r') as f:
    #     model = model_from_json(f.read())

    # Copy model code to build up the model
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    # Utilities
    # Concatenatelayer = Concatenate(axis=2)
    Concatenatelayer1 = Concatenate(axis=-1)
    expand_dim_layer = Lambda(lambda x: K.expand_dims(x,1))
    # expand_dim_layer1 = Lambda(lambda x: K.expand_dims(x,axis=1))
    # get_dim_layer = Lambda(lambda x: x[:,0,0,:,:])
    get_dim_layer1 = Lambda(lambda x: x[:,0,:,:,:])
    # flatten_layer = Flatten()

    # scale_layer = Lambda(lambda x:x/K.sum(x,axis=(1,2)))
    # scale_layer1 = Lambda(lambda x:x/K.sum(x,axis=(1)))
    scale_layer2 = Lambda(lambda x:x/K.sum(x,axis=(1,2,3)))
    # configuration
    kernel_size = cfg.conv_kernel_size
    latent_dim = cfg.latent_dim
    row = cfg.num_row
    col = cfg.num_col
    epochs = 200

    input_shape1 = (cfg.running_length,row,col,1)           # Sample, time, row, col, channel
    input_shape2 = (cfg.running_length,row,col,latent_dim*2)
    input_shape3 = (cfg.running_length,row,col,latent_dim)


    # convLSTM for target past segment average
    encoder_inputs = Input(shape=(cfg.running_length, row, col, 1))
    convlstm_encoder = ConvLSTM2D(filters=latent_dim*2, kernel_size=(kernel_size, kernel_size),
                       input_shape=input_shape1, dropout=cfg.dropout_rate, recurrent_dropout=0.0,
                       stateful=cfg.stateful_across_batch, data_format = 'channels_last',
                       padding='same', return_sequences=True, return_state=True)
    pst_outputs_sqns, pst_state_h0, pst_state_c0 = convlstm_encoder(encoder_inputs)
    states0 = [pst_state_h0, pst_state_c0]
    # print(convlstm_encoder)

    convlstm_encoder1 = ConvLSTM2D(filters=latent_dim, kernel_size=(kernel_size, kernel_size),
                       input_shape=input_shape2, dropout=cfg.dropout_rate, recurrent_dropout=0.0,
                       stateful=cfg.stateful_across_batch, data_format = 'channels_last',
                       padding='same', return_sequences=True, return_state=True)
    pst_outputs_sqns, pst_state_h1, pst_state_c1 = convlstm_encoder1(pst_outputs_sqns)
    states1 = [pst_state_h1, pst_state_c1]

    convlstm_encoder2 = ConvLSTM2D(filters=latent_dim//2, kernel_size=(kernel_size, kernel_size),
                       input_shape=input_shape3, dropout=cfg.dropout_rate, recurrent_dropout=0.0,
                        stateful=cfg.stateful_across_batch, data_format = 'channels_last',
                        padding='same', return_sequences=True, return_state=True)
    # print(pst_outputs_sqns.shape())
    pst_outputs_sqns, pst_state_h2, pst_state_c2 = convlstm_encoder2(pst_outputs_sqns)
    states2 = [pst_state_h2, pst_state_c2]

    # print(pst_outputs_sqns)

    # ###======convLSTM on target future decoder======
    decoder_inputs = Input(shape=(1,row,col,1))   # Only last sequence from encoder
    convlstm_decoder = ConvLSTM2D(filters=latent_dim*2, kernel_size=(kernel_size, kernel_size),
                       input_shape=input_shape1,dropout=cfg.dropout_rate, recurrent_dropout=0.0,
                       stateful=cfg.stateful_across_batch, data_format = 'channels_last',
                       padding='same', return_sequences=True, return_state=True)

    convlstm_decoder1 = ConvLSTM2D(filters=latent_dim, kernel_size=(kernel_size, kernel_size),
                       input_shape=input_shape2,dropout=cfg.dropout_rate, recurrent_dropout=0.0,
                       stateful=cfg.stateful_across_batch, data_format = 'channels_last',
                       padding='same', return_sequences=True, return_state=True)

    convlstm_decoder2 = ConvLSTM2D(filters=latent_dim//2, kernel_size=(kernel_size, kernel_size),
                       input_shape=input_shape3,dropout=cfg.dropout_rate, recurrent_dropout=0.0,
                       stateful=cfg.stateful_across_batch, data_format = 'channels_last',
                       padding='same', return_sequences=True, return_state=True)


    # ### 2D conv
    pred_conv_conv = Conv2D(filters=64, kernel_size=(kernel_size,kernel_size), padding='same',
        activation='relu', use_bias=True, kernel_initializer='glorot_uniform')
    pred_conv_conv1 = Conv2D(filters=128, kernel_size=(kernel_size,kernel_size), padding='same',
        activation='relu', use_bias=True, kernel_initializer='glorot_uniform')
    pred_conv_conv2 = Conv2D(filters=1, kernel_size=(kernel_size,kernel_size), padding='same',   # filters=fps
        activation='relu', use_bias=True, kernel_initializer='glorot_uniform')

    # 2D conv for other users' gt
    others_conv0 = Conv2D(filters=16, kernel_size=(kernel_size,kernel_size), padding='same',
        activation='relu', use_bias=True, kernel_initializer='glorot_uniform')
    others_conv1 = Conv2D(filters=32, kernel_size=(kernel_size,kernel_size), padding='same',
        activation='relu', use_bias=True, kernel_initializer='glorot_uniform')
    # others_conv2 = Conv2D(filters=1, kernel_size=(kernel_size,kernel_size), padding='same',   # filters=fps
    #     activation='relu', use_bias=True, kernel_initializer='glorot_uniform')

    # 2D conv for other users' var gt
    others_var_conv0 = Conv2D(filters=16, kernel_size=(kernel_size,kernel_size), padding='same',
        activation='relu', use_bias=True, kernel_initializer='glorot_uniform')
    others_var_conv1 = Conv2D(filters=32, kernel_size=(kernel_size,kernel_size), padding='same',
        activation='relu', use_bias=True, kernel_initializer='glorot_uniform')
    # others_var_conv2 = Conv2D(filters=1, kernel_size=(kernel_size,kernel_size), padding='same',   # filters=fps
    #     activation='relu', use_bias=True, kernel_initializer='glorot_uniform')

    # num_conv1 = Conv2D(filters=32, kernel_size=(kernel_size,kernel_size), padding='same',
        # activation='relu', use_bias=True, kernel_initializer='glorot_uniform')

    pred_interval_conv1 = Conv2D(filters=4, kernel_size=(kernel_size,kernel_size), padding='same',
        activation='relu', use_bias=True, kernel_initializer='glorot_uniform')

    # Dense for number of users
    # other_user_dense = Dense(8, input_shape=(1,),activation='relu')

    # output final map for next timestamp
    # final_output = Dense(row*col, input_shape=(2*row*col+16,),activation='softmax')
    # final_reshape = Reshape((row,col,1), input_shape=(row*col,))

    # bnlayer0 = BatchNormalization(axis=-1,center=True,scale=True)
    # bnlayer1 = BatchNormalization(axis=-1,center=True,scale=True)
    # bnlayer2 = BatchNormalization(axis=-1,center=True,scale=True)

    # 
    all_outputs= []
    inputs = decoder_inputs
    other_inputs = Input(shape=(cfg.predict_step, row, col, 1))
    other_inputs_var = Input(shape=(cfg.predict_step, row, col, 1))
    num_other = Input(shape=(cfg.predict_step, row, col, 1))
    pred_inteval = Input(shape=(cfg.predict_step, row, col, 1))

    # num_users = Input(shape=(1))
    for time_ind in range(cfg.predict_step):
        # print('k0', inputs.shape)
        fut_outputs_sqns0, fut_state_h, fut_state_c = convlstm_decoder([inputs]+states0)
        states0 = [fut_state_h, fut_state_c]
        fut_outputs_sqns1, fut_state_h, fut_state_c = convlstm_decoder1([fut_outputs_sqns0]+states1)
        states1 = [fut_state_h, fut_state_c]
        fut_outputs_sqns2, fut_state_h, fut_state_c = convlstm_decoder2([fut_outputs_sqns1]+states2)
        states2 = [fut_state_h, fut_state_c]

        fut_outputs_sqns = Concatenatelayer1([fut_outputs_sqns0,fut_outputs_sqns1,fut_outputs_sqns2])
        print(fut_outputs_sqns.shape)
        # Concatenate ground truth and var

        groud_truth_map = other_inputs[:,time_ind,:,:,:]
        # print(groud_truth_map.shape)
        groud_truth_map_var = other_inputs_var[:,time_ind,:,:,:]
        num_other_gt = num_other[:,time_ind,:,:,:]
        pred_interval_gt = pred_inteval[:,time_ind,:,:,:]

        groud_truth_map = others_conv0(groud_truth_map)
        groud_truth_map = others_conv1(groud_truth_map)

        groud_truth_map_var = others_var_conv0(groud_truth_map_var)
        groud_truth_map_var = others_var_conv1(groud_truth_map_var)    
        # print('l', groud_truth_map_var.shape)
        # print(groud_truth_map_var.shape)
        num_other_map = num_conv1(num_other_gt)
        # print('l', num_other_map.shape)
        pre_interval_map = pred_interval_conv1(pred_interval_gt)
        # print('l', pre_interval_map.shape)

        groud_truth_map = expand_dim_layer(groud_truth_map)
        groud_truth_map_var = expand_dim_layer(groud_truth_map_var)
        num_other_map = expand_dim_layer(num_other_map)
        pre_interval_map = expand_dim_layer(pre_interval_map)

        fut_outputs_sqns = Concatenatelayer1([fut_outputs_sqns, groud_truth_map, groud_truth_map_var, num_other_map, pre_interval_map])
        print(fut_outputs_sqns.shape)
        # fut_outputs_sqns = bnlayer(fut_outputs_sqns)
        ### predict others' future
        outputs = get_dim_layer1(fut_outputs_sqns)
        # print('k1', outputs.shape)
        outputs = pred_conv_conv(outputs)
        # print('k1', outputs.shape)
        # outputs = bnlayer0(outputs)
        outputs = pred_conv_conv1(outputs)
        # outputs = bnlayer1(outputs)
        outputs = pred_conv_conv2(outputs)

        # print('f3', outputs.shape)
        outputs = scale_layer2(outputs)
        outputs = expand_dim_layer(outputs)
        # print('final', outputs.shape)
        inputs = outputs

        all_outputs.append(outputs)

    # Concatenate all predictions
    # print('all_outputs', all_outputs)
    decoder_outputs = Lambda(lambda x: K.concatenate(x, axis=1))(all_outputs)
    final_shape = Reshape((cfg.predict_step,row*col), input_shape=(cfg.predict_step,row,col,1))
    decoder_outputs = final_shape(decoder_outputs)

    print('encoder_input', encoder_inputs.shape)
    print('decoder_input:', decoder_inputs.shape)
    print('other_inputs: ', other_inputs.shape)
    print('other_inputs_var: ', other_inputs_var.shape)

    model = Model([encoder_inputs,decoder_inputs,other_inputs,other_inputs_var,num_other,pred_inteval],decoder_outputs)

    model.load_weights(model_path)


    # Directly load model might cause version issue
    # model = load_model(model_path)

    return model
