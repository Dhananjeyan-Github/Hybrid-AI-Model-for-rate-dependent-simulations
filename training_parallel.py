#%%
import os
import random
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers, losses, models, layers
import joblib

from nns_traj_double import pod_deepnet as PDN
from utils_parallel import prep


SEED = 24
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  print(gpu)
  tf.config.experimental.set_memory_growth(gpu, True)
tf.config.set_visible_devices(gpus, 'GPU')
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

model_name = os.path.expanduser('./models/tf_2025_lstsq_8')
if not os.path.exists(model_name):
    os.makedirs(model_name)
sdir = os.path.expanduser(model_name)

boundary_condition=os.path.expanduser('./data/boundary_condition.npz')
# pp = prep(boundary_condition, n_trajectories=900)
# joblib.dump(pp, sdir + 'pp')
pp = joblib.load('./data/tf_2025_lstsq_8pp')

x_train, x_test, x_val, x_mean, x_std = pp.scale()
print(x_train.shape)
print(x_test.shape)
print(x_val.shape)
time_train, time_test, time_val = pp.load_time(boundary_condition)

x_train = np.concatenate([time_train.reshape(time_train.shape[0], time_train.shape[-1], 1), x_train], axis=-1)
x_test = np.concatenate([time_test.reshape(time_test.shape[0], time_test.shape[-1], 1), x_test], axis=-1)
x_val = np.concatenate([time_val.reshape(time_val.shape[0], time_val.shape[-1], 1), x_val], axis=-1)

file_name_u= os.path.expanduser('./data/def_viscoelas.npz')
u_train, u_test, u_val, u_mean, u_std = pp.load_local(file_name_u)

file_name_q= os.path.expanduser('./data/intvar_viscoelas.npz')
q_train, q_test, q_val, q_mean, q_std = pp.load_local(file_name_q)

u_train=np.concatenate(u_train)
# modes_u_train, sigma_u_train, VT_u_train = np.linalg.svd(u_train.T, full_matrices=False)
modes_u_train = joblib.load('./data/tf_2025_lstsq_8U')
r_1 = 16
modes_u_train = modes_u_train[:, :r_1]
print('u_train_modes')
print(modes_u_train.shape)
u_train = u_train.reshape(u_train.shape[0]//10, 10, -1)

q_train=np.concatenate(q_train)
# modes_q_train, sigma_q_train, VT_q_train = np.linalg.svd(q_train.T, full_matrices=False)
modes_q_train = joblib.load('./data/tf_2025_lstsq_8Q')
r_2 = 8
new_modes_q_train = modes_q_train[:, :4]
modes_q_train = modes_q_train[:, :r_2]


b_q_train = tf.transpose(tf.linalg.lstsq(modes_q_train, tf.transpose(q_train)))
new_b_q_train = tf.transpose(tf.linalg.lstsq(new_modes_q_train, tf.transpose(q_train)))
new_b_q_train = tf.reshape(new_b_q_train, [new_b_q_train.shape[0]//10, 10, 4])
b_q_train = tf.reshape(b_q_train, [b_q_train.shape[0]//10, 10, r_2])


q_train = q_train.reshape(q_train.shape[0]//10, 10, -1)
b_q_train_input = np.zeros_like(b_q_train)
b_q_train_input[:, 1:, :] = b_q_train[:, :-1, :]
new_b_q_train_input = np.zeros_like(new_b_q_train)
new_b_q_train_input[:, 1:, :] = new_b_q_train[:, :-1, :]

q_test=np.concatenate(q_test)
r_3 = 8
b_q_test = tf.transpose(tf.linalg.lstsq(modes_q_train, tf.transpose(q_test)))
new_b_q_test = tf.transpose(tf.linalg.lstsq(new_modes_q_train, tf.transpose(q_test)))

new_b_q_test = tf.reshape(new_b_q_test, [new_b_q_test.shape[0]//10, 10, 4])
b_q_test = tf.reshape(b_q_test, [b_q_test.shape[0]//10, 10, r_3])
q_test = q_test.reshape(q_test.shape[0]//10, 10, -1)
b_q_test_input = np.zeros_like(b_q_test)
b_q_test_input[:, 1:, :] = b_q_test[:, :-1, :]
new_b_q_test_input = np.zeros_like(new_b_q_test)
new_b_q_test_input[:, 1:, :] = new_b_q_test[:, :-1, :]


array_list = [x_train, x_test, x_val, u_train, u_test, u_val, q_train, q_test, q_val, b_q_train, b_q_train_input, b_q_test_input, new_b_q_train_input, new_b_q_test_input]
array_list = [arr[:, :-1, :] for arr in array_list]
x_train, x_test, x_val, u_train, u_test, u_val, q_train, q_test, q_val, b_q_train, b_q_train_input, b_q_test_input, new_b_q_train_input, new_b_q_test_input = array_list
concatenated_arrays = [np.concatenate(arr) for arr in array_list]
x_train, x_test, x_val, u_train, u_test, u_val, q_train, q_test, q_val, b_q_train, b_q_train_input, b_q_test_input, new_b_q_train_input, new_b_q_test_input = concatenated_arrays

b_u_train_ref = tf.transpose(tf.linalg.lstsq(modes_u_train, tf.transpose(u_train)))
b_u_test_ref = tf.transpose(tf.linalg.lstsq(modes_u_train, tf.transpose(u_test)))
x_train = np.concatenate((x_train, new_b_q_train_input), axis=-1)
x_test = np.concatenate((x_test, new_b_q_test_input), axis=-1)


def gen_models_double(nf, nv, act, nn, nl):
    inp = layers.Input((nf,))
    x = layers.Dense(nn, activation = act)(inp)

    for _ in range(nl - 1):
        x = layers.Dense(nn, activation = act)(x)    
    out = layers.Dense(nv)(x)

    model = models.Model(inp, out)
    return model

m = x_train.shape[-1]
act_1 = tf.keras.activations.swish
n_batches = 64
batch_size = int(len(x_train) / n_batches)
initial_learning_rate = 1e-3
lr_schedule = optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=1000 * n_batches,
    decay_rate=0.5,
    staircase=False)


branch_1 = gen_models_double(m, r_1, act_1, 128, 2)
model = PDN(branch_1)
optimizer_1 = optimizers.Adam(lr_schedule)
loss_1 = losses.MeanSquaredError()
model.compile(
    optimizer_1=optimizer_1,
    loss_1=loss_1,
)


hist = model.fit([x_train], [b_u_train_ref], 
          epochs=10000, 
          batch_size = batch_size, 
          validation_data=([x_test], [b_u_test_ref]), 
          verbose = 2)
hist = hist.history

model.save(sdir + 'model_lstsq_8_4', save_format='tf')
from tensorflow.keras.models import load_model
model = load_model(sdir + 'model_lstsq_8_4',
    custom_objects={'PDN': PDN})


b_u_test_pred = model.predict(x_test)
u_pred = modes_u_train @ tf.transpose(b_u_test_pred)
u_pred = tf.transpose(u_pred)
u_test = (u_test*u_std) + u_mean
u_pred = (u_pred*u_std) + u_mean

print('u_error')
print(np.mean(np.linalg.norm(u_test - u_pred, axis=1) / np.linalg.norm(u_test, axis=1)))


