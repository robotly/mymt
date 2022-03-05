from tensorflow.keras.layers import Conv2D, MaxPool2D, PReLU, Softmax,Reshape,Flatten,Dense
from tensorflow.keras import Input,losses
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam,SGD
#from tensorflow.keras.utils import to_categorical
import numpy as np
import _pickle as pickle
import tensorflow as tf
import random

tf.enable_eager_execution()
with open(r'..\prepare_data\imdb\24\positive_v1.imdb', 'rb') as fid:
    positive = pickle.load(fid)
with open(r'..\prepare_data\imdb\24\part_v1.imdb', 'rb') as fid:
    part = pickle.load(fid)
with open(r'..\prepare_data\imdb\24\negative_v2.imdb', 'rb') as fid:
    negative = pickle.load(fid)
with open(r'..\prepare_data\imdb\24\pts_v1.imdb', 'rb') as fid:
    pts = pickle.load(fid)
data=[]
data.extend(positive)
data.extend(part)
data.extend(negative)
data.extend(pts)
random.shuffle(data)
length=len(data)
batch_size = 64
steps=length//batch_size

input = Input(shape=(24, 24, 3))
r_layer = Conv2D(28, kernel_size=(3, 3), strides=(1, 1), padding="valid", name='conv1')(input)
r_layer = PReLU(shared_axes=[1, 2], name='prelu1')(r_layer)
r_layer = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding="same")(r_layer)

r_layer = Conv2D(48, kernel_size=(3, 3), strides=(1, 1), padding="valid", name='conv2')(r_layer)
r_layer = PReLU(shared_axes=[1, 2], name='prelu2')(r_layer)
r_layer = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding="valid")(r_layer)

r_layer = Conv2D(64, kernel_size=(2, 2), strides=(1, 1), padding="valid", name='conv3')(r_layer)
r_layer = PReLU(shared_axes=[1, 2], name='prelu3')(r_layer)
r_layer = Flatten()(r_layer)
r_layer = Dense(128,name='dense1')(r_layer)
r_layer = PReLU(name='prelu4')(r_layer)

r_layer_out1 = Dense(2,name='classifier')(r_layer)

r_layer_out2 = Dense(4,name='bbox')(r_layer)

r_layer_out3 = Dense(10,name='landmark')(r_layer)
model = Model([input], [r_layer_out1,r_layer_out2,r_layer_out3])


optimizer=Adam()
cc=losses.CategoricalCrossentropy(from_logits=True)
for epoch in range(10):
    for step in range(steps):
        loss_all = 0
        with tf.GradientTape() as tape:
            for i in range(0+step*batch_size,(step+1)*batch_size):
                x=data[i]
                im=np.expand_dims(np.transpose(x[0],(1,0,2)),axis=0)
                out = model(im.astype(np.float32))
                if type(x[1])==int:
                    if x[1]==1:
                        one_hot = np.array([0.,1.],dtype=np.float32)
                        loss = cc(one_hot,out[0][0])+0.5*tf.math.reduce_sum(tf.math.squared_difference(x[2],out[1][0]))
                    elif x[1]==0:
                        one_hot = np.array([1., 0.], dtype=np.float32)
                        loss = cc(one_hot, out[0][0])
                    else:
                        loss = 0.5 * tf.math.reduce_sum(tf.math.squared_difference(x[2], out[1][0]))
                else:
                    loss = 0.5 * tf.math.reduce_sum(tf.math.squared_difference(x[1], out[2][0]))
                loss_all+=loss
            loss_all=loss_all/batch_size
        if step % 500 == 0:
            print(epoch, step, loss_all.numpy())
        grads = tape.gradient(loss_all, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

model_s = Model([input], [r_layer_out1, r_layer_out2])
model_s.save_weights('rmodel_v2_transpose.h5')
