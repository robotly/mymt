from tensorflow.keras.layers import Conv2D, MaxPool2D, PReLU, Softmax,Reshape
from tensorflow.keras import Input,losses
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam,SGD
#from tensorflow.keras.utils import to_categorical
import numpy as np
import _pickle as pickle
import tensorflow as tf
import random

tf.enable_eager_execution()
with open(r'..\prepare_data\imdb\12\positive.imdb', 'rb') as fid:
    positive = pickle.load(fid)
with open(r'..\prepare_data\imdb\12\part.imdb', 'rb') as fid:
    part = pickle.load(fid)
with open(r'..\prepare_data\imdb\12\negative.imdb', 'rb') as fid:
    negative = pickle.load(fid)
with open(r'..\prepare_data\imdb\12\pts_v1.imdb', 'rb') as fid:
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

# ims_pos = []
# cls_pos = []
# roi_pos=[]
# ims_part = []
# roi_part = []
# ims_neg = []
# cls_neg = []
# ims_pts=[]
# reg_pts=[]
# for dataset in positive:
    # ims_pos.append(dataset[0])
    # cls_pos.append(dataset[1])
    # roi_pos.append(dataset[2])
# for dataset in part:
    # ims_part.append(dataset[0])
    # roi_part.append(dataset[2])
# for dataset in negative:
    # ims_neg.append(dataset[0])
    # cls_neg.append(dataset[1])
# for dataset in pts:
    # ims_pts.append(dataset[0])
    # reg_pts.append(dataset[1])

# ims_pos = np.array(ims_pos)
# ims_neg = np.array(ims_neg)
# ims_part = np.array(ims_part)
# cls_pos = np.array(cls_pos)
# cls_neg = np.array(cls_neg)
# roi_pos = np.array(roi_pos)
# roi_part = np.array(roi_part)
# ims_pts = np.array(ims_pts)
# reg_pts = np.array(reg_pts)

#one_hot_pos = to_categorical(cls_pos, num_classes=2)
#one_hot_neg = to_categorical(cls_neg, num_classes=2)

# Pnet
input = Input(shape=(12, 12, 3))
xx = Conv2D(10, (3, 3), strides=(1,1), padding='valid', name='conv1')(input)
xx = PReLU(shared_axes=[1, 2], name='prelu1')(xx)
xx = MaxPool2D(pool_size=(2,2),padding='same')(xx)
xx = Conv2D(16, (3, 3), strides=(1,1), padding='valid', name='conv2')(xx)
xx = PReLU(shared_axes=[1, 2], name='prelu2')(xx)
xx = Conv2D(32, (3, 3), strides=(1,1), padding='valid', name='conv3')(xx)
xx = PReLU(shared_axes=[1, 2], name='prelu3')(xx)
classifier = Conv2D(2, (1, 1),strides=(1,1), name='classifier')(xx)
classifier1 = Reshape((2,), name='classifier1')(classifier)
bbox_regress = Conv2D(4, (1, 1),strides=(1,1), name='bbox')(xx)
bbox_regress1 = Reshape((4,), name='bbox1')(bbox_regress)
landmark_regress = Conv2D(10, (1, 1),strides=(1,1), name='landmark')(xx)
landmark_regress1 = Reshape((10,), name='landmark1')(landmark_regress)
model = Model([input], [classifier1,bbox_regress1,landmark_regress1])

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

# batch_size = 64
# randx=[0,1,2,3]*20
# random.shuffle(randx)
# for i_train in range(80):
    # print('currently in training macro cycle: ', i_train)
    # if 0 == randx[i_train]:
        # model = Model([input], [classifier1,bbox_regress1])
        # model.compile(
            # optimizer=RMSprop(1e-5),
            # loss={
                # "classifier1": losses.CategoricalCrossentropy(from_logits=True),
                # "bbox1": losses.mean_squared_error,
            # },
            # loss_weights=[1.0, 0.5],
        # )
        # model.fit(ims_pos, {"classifier1":one_hot_pos,"bbox1":roi_pos},batch_size=batch_size)

    # if 1 == randx[i_train]:
        # model = Model([input], [bbox_regress1])
        # model.compile(
            # optimizer=RMSprop(1e-5),
            # loss={
                # "bbox1": losses.mean_squared_error,
            # },
            # loss_weights=[0.5],
        # )
        # model.fit(ims_part, {"bbox1":roi_part}, batch_size=batch_size, epochs=1)

    # if 2 == randx[i_train]:
        # model = Model([input], [classifier1])
        # model.compile(
            # optimizer=RMSprop(1e-5),
            # loss={
                # "classifier1": losses.CategoricalCrossentropy(from_logits=True),
            # },
        # )
        # model.fit(ims_neg, {"classifier1":one_hot_neg}, batch_size=batch_size, epochs=1)

    # if 3 == randx[i_train]:
        # model = Model([input], [landmark_regress1])
        # model.compile(
            # optimizer=RMSprop(1e-5),
            # loss={
                # "landmark1": losses.mean_squared_error,
            # },
            # loss_weights=[0.5],
        # )
        # model.fit(ims_pts, {"landmark1":reg_pts}, batch_size=batch_size, epochs=1)

model_s = Model([input], [classifier, bbox_regress])
model_s.save_weights('pmodel_v1_transpose_RGB.h5')
