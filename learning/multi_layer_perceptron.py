#%%
import sys,math,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from icecream import ic

#%%
if len(sys.argv) == 2:
    filename = sys.argv[1]
else:
    filename = "sample_training_data.txt"
org_training = np.loadtxt(filename)
ic(filename, np.shape(org_training))

#%%
# format of training data
# 0-12 : input parameters
#    N, p_tri, p_r, p_nd, p_ld,
#    aging, w_th, w_r, q, F,
#    alpha, t_max, seed
# 13-22: outputs
#    <k>, stddev(k), <w>, pcc(knn), cc
#    pcc(ck), O, pcc(Ow), comm_size, comm_degeneracy
training = org_training[ org_training[:,13] > 0,: ]
ic(np.shape(training), training[0])
#%%
# h_params = {
#         "ycol": 0,
#         "units1": 30,
#         "units2": 10,
#         "units3": 0,
#         "activation": "relu",  # relu, tanh, sigmoid
#         "lr": 0.001,
#         "batch_size": 200,
#         "epochs": 2000
#         }
def load_input_json(json_path):
    with open(json_path) as f:
        return json.load(f)
h_params = load_input_json('_input.json')
ic(h_params)

#%%
# parameters:
x = np.log10(training[:,1:6])
x = np.concatenate([training[:,0:1],x,training[:,7:12]], axis=1 )
ic(np.shape(x))

#%%
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

#%%
def scale_input(data):
    #from sklearn.preprocessing import StandardScaler
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    #scaler = StandardScaler()
    scaled = scaler.fit_transform(data)
    return (scaler, scaled)

scaler, x_train = scale_input(x)
ic(scaler.min_, scaler.scale_, x, x_train)

#%%
y_train = training[:,13 + h_params['ycol']]
if h_params['ycol'] == 2:
    y_train = np.log10(y_train)  # take logarithm for link weight
elif h_params['ycol'] == 8 or h_params['ycol'] == 9:
    b = y_train < 10.0
    y_train = y_train[b]
    x_train = x_train[b]

ic(y_train.shape, y_train, y_train.min(), y_train.max())

#%%
def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(h_params["h_units"], input_dim=np.shape(x)[1], kernel_initializer='normal', activation=h_params["activation"]))
    model.add(Dense(1, kernel_initializer='normal'))
    opt = Adam(lr=h_params["lr"], beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    # Compile model
    model.compile(loss='mean_squared_error', optimizer=opt)
    return model

def larger_model():
    # create model
    model = Sequential()
    model.add(Dense(h_params["units1"], input_dim=np.shape(x)[1], kernel_initializer='normal', activation=h_params["activation"]))
    if h_params["units2"] > 0:
        model.add(Dense(h_params["units2"], kernel_initializer='normal', activation=h_params["activation"]))
    if h_params["units3"] > 0:
        model.add(Dense(h_params["units3"], kernel_initializer='normal', activation=h_params["activation"]))
    model.add(Dense(1, kernel_initializer='normal'))
    opt = Adam(lr=h_params["lr"], beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    # Compile model
    model.compile(loss='mean_squared_error', optimizer=opt)
    return model

#m = baseline_model()
m = larger_model()
es = EarlyStopping(monitor='val_loss', patience=50, verbose=0, mode='auto')
h = m.fit(x_train, y_train, batch_size=h_params["batch_size"], epochs=h_params["epochs"], callbacks=[es], verbose=0, validation_split=0.05)
ic(m.predict(x_train)[:,0], y_train)
m.predict(x_train)

#%%
def plot_history(h):
    import matplotlib.pyplot as plt
    plt.clf()
    plt.plot(h['loss'][:])
    plt.plot(h['val_loss'][:])
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['loss', 'val_loss'], loc='best')
    plt.ylim(ymin=0)
    plt.savefig('loss.png')

def output_results(h):
    min_val_loss = min(h['val_loss'])
    min_loss = min(h['loss'])
    with open("_output.json", 'w') as f:
        json.dump({"min_val_loss": min_val_loss, "min_loss": min_loss}, f)

print("min: ", min(h.history['val_loss']))
output_results(h.history)
plot_history(h.history)

m.save('model.h5')


# %%
