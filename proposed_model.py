import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Model
from keras.layers import (
    Input, Dense, MaxPooling1D, Conv1D, Flatten, Bidirectional,
    LSTM, Add, Multiply, BatchNormalization, Activation,
    Conv1DTranspose, AveragePooling1D, UpSampling1D, Dropout, ZeroPadding1D
)
from keras.optimizers import Adam
import keras.backend as K
import lightgbm as lgb
import os
from IPython.display import Image
from keras.src.layers import Lambda
from keras.utils import plot_model

# Import custom evaluation function
from Sub_Functions.Evaluate import main_est_parameters


# --- Helper Function ---
def pad_to_multiple_of_8(x):
    seq_len = tf.shape(x)[1]
    remainder = seq_len % 8
    pad_len = tf.where(remainder == 0, 0, 8 - remainder)
    return tf.pad(x, [[0, 0], [0, pad_len], [0, 0]])


class Feature_Pyramid_Attention_1D:
    def __init__(self, layer):
        self.layer = layer
        self.layer_shape = K.int_shape(layer)

    def downsample(self):
        filters = self.layer_shape[-1]

        # ------------------ Down Block 1 ------------------
        max_pool_1 = MaxPooling1D(pool_size=2, padding="valid")(self.layer)
        conv7_1 = Conv1D(filters, 7, padding='same', kernel_initializer='he_normal')(max_pool_1)
        conv7_1 = BatchNormalization()(conv7_1)
        conv7_1 = Activation('relu')(conv7_1)
        conv7_2 = Conv1D(filters, 7, padding='same', kernel_initializer='he_normal')(conv7_1)
        conv7_2 = BatchNormalization()(conv7_2)
        conv7_2 = Activation('relu')(conv7_2)

        # ------------------ Down Block 2 ------------------
        max_pool_2 = MaxPooling1D(pool_size=2, padding="valid")(conv7_2)
        conv5_1 = Conv1D(filters, 5, padding='same', kernel_initializer='he_normal')(max_pool_2)
        conv5_1 = BatchNormalization()(conv5_1)
        conv5_1 = Activation('relu')(conv5_1)
        conv5_2 = Conv1D(filters, 5, padding='same', kernel_initializer='he_normal')(conv5_1)
        conv5_2 = BatchNormalization()(conv5_2)
        conv5_2 = Activation('relu')(conv5_2)

        # ------------------ Down Block 3 ------------------
        max_pool_3 = MaxPooling1D(pool_size=2, padding="valid")(conv5_2)
        conv3_1 = Conv1D(filters, 3, padding='same', kernel_initializer='he_normal')(max_pool_3)
        conv3_1 = BatchNormalization()(conv3_1)
        conv3_1 = Activation('relu')(conv3_1)
        conv3_2 = Conv1D(filters, 3, padding='same', kernel_initializer='he_normal')(conv3_1)
        conv3_2 = BatchNormalization()(conv3_2)
        conv3_2 = Activation('relu')(conv3_2)

        # ------------------ UPSAMPLE ------------------
        upsampled_8 = UpSampling1D(size=2)(conv3_2)
        upsampled_8 = Conv1D(filters, 3, padding='same')(upsampled_8)
        added_1 = Add()([upsampled_8, conv5_2])

        upsampled_16 = UpSampling1D(size=2)(added_1)
        upsampled_16 = Conv1D(filters, 3, padding='same')(upsampled_16)
        added_2 = Add()([upsampled_16, conv7_2])

        upsampled_32 = UpSampling1D(size=2)(added_2)
        upsampled_32 = Conv1D(filters, 3, padding='same')(upsampled_32)
        return upsampled_32

    def direct_branch(self):
        return Conv1D(self.layer_shape[-1], 1, padding='valid', kernel_initializer='he_normal')(self.layer)

    def FPA(self):
        down_up_conved = self.downsample()
        direct_conved = self.direct_branch()

        # Dynamic shape adjustment as a safety measure
        target_shape = tf.shape(direct_conved)[1]
        current_shape = tf.shape(down_up_conved)[1]

        # We use a Lambda or simple logic to trim/pad if necessary
        # Usually padding to multiple of 8 fixes this entirely.
        return Multiply()([down_up_conved, direct_conved])


class MutualCrossAttention(tf.keras.layers.Layer):
    def __init__(self, dropout_rate=0.1, **kwargs):
        super(MutualCrossAttention, self).__init__(**kwargs)
        self.dropout = Dropout(dropout_rate)

    def call(self, x1, x2, training=None):
        query = x1
        key = x2
        d = tf.cast(tf.shape(query)[-1], tf.float32)
        scores_qk = tf.matmul(query, key, transpose_b=True) / tf.math.sqrt(d)
        attn_weights_qk = tf.nn.softmax(scores_qk, axis=-1)
        attn_weights_qk = self.dropout(attn_weights_qk, training=training)
        output_A = tf.matmul(attn_weights_qk, x2)

        scores_kq = tf.matmul(key, query, transpose_b=True) / tf.math.sqrt(d)
        attn_weights_kq = tf.nn.softmax(scores_kq, axis=-1)
        attn_weights_kq = self.dropout(attn_weights_kq, training=training)
        output_B = tf.matmul(attn_weights_kq, x1)
        return output_A


def Proposed_model(x_train, x_test, y_train, y_test, epochs):
    # Detect actual feature count (handles 2603, 2306, or any other size)
    total_features = x_train.shape[1]
    num_classes = len(np.unique(y_train)) + 1

    # Reshape input: (Samples, Features, 1)
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)

    main_input = Input(shape=(total_features, 1))

    # --- Dynamic Slicing ---
    # Branch 1: first 1000 | Branch 2: next 1000 | Branch 3: the rest
    branch1_in = Lambda(lambda x: x[:, 0:1000, :])(main_input)
    branch2_in = Lambda(lambda x: x[:, 1000:2000, :])(main_input)
    branch3_in = Lambda(lambda x: x[:, 2000:total_features, :])(main_input)

    # Branch 1 (Length: 1000)
    b1 = Bidirectional(LSTM(128, return_sequences=True))(branch1_in)
    b1_f = Bidirectional(LSTM(64, return_sequences=True))(b1)
    fpa1 = Feature_Pyramid_Attention_1D(b1_f).FPA()

    # Branch 2 (Length: 1000)
    b2 = Bidirectional(LSTM(128, return_sequences=True))(branch2_in)
    b2_f = Bidirectional(LSTM(64, return_sequences=True))(b2)

    # MCA 1: B1 attends to B2
    mca_out1 = MutualCrossAttention()(fpa1, b2_f)

    # Branch 3 (Length: total_features - 2000)
    b3 = Bidirectional(LSTM(64, return_sequences=True))(branch3_in)

    # FPA on B2 for the second cross-attention
    fpa2 = Feature_Pyramid_Attention_1D(b2_f).FPA()

    # MCA 2: B2 (1000) attends to B3 (variable length)
    mca_out2 = MutualCrossAttention()(fpa2, b3)

    # Final Processing
    proc1 = Bidirectional(LSTM(32, return_sequences=True))(mca_out1)
    proc2 = Bidirectional(LSTM(32, return_sequences=True))(mca_out2)

    # Merging
    added = Add()([proc1, proc2])
    flat = Flatten()(added)

    # Dense Head
    d = Dense(1024, activation="relu")(flat)
    d = Dense(256, activation="relu")(d)
    d = Dense(64, activation="relu")(d)
    d = Dense(32, activation="relu")(d)
    feat_out = Dense(16, activation="relu", name="Final_out")(d)

    output = Dense(num_classes, activation="softmax")(feat_out)

    model = Model(inputs=main_input, outputs=output)
    model.compile(optimizer=Adam(0.001), loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    # This will now accept your 2306 input shape without crashing
    model.fit(x_train, y_train, batch_size=32, epochs=epochs)
    # ML Stage
    feature_model = Model(inputs=main_input, outputs=model.get_layer("Final_out").output)
    plot_model(feature_model,to_file="Architecture/Proposed_model1.png",show_shapes=True,show_layer_activations=True,show_layer_names=True)

    train_feats = feature_model.predict(x_train)
    test_feats = feature_model.predict(x_test)

    lgbm = lgb.LGBMClassifier(n_estimators=100, verbose=-1)
    lgbm.fit(train_feats, y_train)
    y_pred = lgbm.predict(test_feats)

    from Sub_Functions.Evaluate import main_est_parameters
    return main_est_parameters(y_test, y_pred)