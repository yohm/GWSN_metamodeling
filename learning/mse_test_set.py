#%%
import sys,math,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras.models import load_model
from icecream import ic

#%%
model_ycol = [
    {
        "model": '/Users/murase/work/oacis/public/Result_development/60c01079709711a8751ac2e4/60c2ef6b709711a8751ac332/60c2ef6b709711a8751ac33c/model.h5',
        "ycol": 0,
        "name": "k"
    },
    {
        "model": '/Users/murase/work/oacis/public/Result_development/60c01079709711a8751ac2e4/60c2ef6b709711a8751ac335/60c2ef6b709711a8751ac33f/model.h5',
        "ycol": 3,
        "name": "rho_kk"
    },
    {
        "model": '/Users/murase/work/oacis/public/Result_development/60c01079709711a8751ac2e4/60c2ef6b709711a8751ac334/60c835fe709711a8751ac41f/model.h5',
        "ycol": 2,
        "name": "w"
    },
    {
        "model": '/Users/murase/work/oacis/public/Result_development/60c01079709711a8751ac2e4/60c2ef6b709711a8751ac336/60c2ef6b709711a8751ac340/model.h5',
        "ycol": 4,
        "name": "cc"
    },
    {
        "model": '/Users/murase/work/oacis/public/Result_development/60c01079709711a8751ac2e4/60c2ef6b709711a8751ac337/60c2ef6b709711a8751ac341/model.h5',
        "ycol": 5,
        "name": "rho_ck"
    },
    {
        "model": '/Users/murase/work/oacis/public/Result_development/60c01079709711a8751ac2e4/60c2ef6b709711a8751ac338/60c2ef6b709711a8751ac342/model.h5',
        "ycol": 6,
        "name": "overlap"
    },
    {
        "model": '/Users/murase/work/oacis/public/Result_development/60c01079709711a8751ac2e4/60c2ef6b709711a8751ac339/60c2ef6b709711a8751ac343/model.h5',
        "ycol": 7,
        "name": "rho_ow"
    },
    {
        "model": '/Users/murase/work/oacis/public/Result_development/60c01079709711a8751ac2e4/60c2ef6b709711a8751ac33a/60d1f5d570971125401e9872/model.h5',
        "ycol": 8,
        "name": "perc_a"
    },
    {
        "model": '/Users/murase/work/oacis/public/Result_development/60c01079709711a8751ac2e4/60c2ef6b709711a8751ac33b/60d1f5dd70971125401e987b/model.h5',
        "ycol": 9,
        "name": "perc_d"
    }
]

#%%
if len(sys.argv) == 2:
    test_set_path = sys.argv[1]
else:
    test_set_path = "sample_training_data.txt"

ic(test_set_path)

#%%
org_test_set = np.loadtxt(test_set_path)

#%%
# format of training data
# 0-12 : input parameters
#    N, p_tri, p_r, p_nd, p_ld,
#    aging, w_th, w_r, q, F,
#    alpha, t_max, seed
# 13-22: outputs
#    <k>, stddev(k), <w>, pcc(knn), cc
#    pcc(ck), O, pcc(Ow), comm_size, comm_degeneracy
test_set = org_test_set[ org_test_set[:,13] > 0,: ]
ic(np.shape(test_set), test_set[0])
#%%

#%%
# parameters:
x = np.log10(test_set[:,1:6])
x = np.concatenate([test_set[:,0:1],x,test_set[:,7:12]], axis=1 )
ic(np.shape(x))

#%%
scale = np.array([2.50000000e-04, 3.33335264e-01, 5.00002063e-01, 5.00001303e-01, 5.00001520e-01, 3.33335794e-01, 5.00000359e-01, 1.11111111e-01, 1.11111111e-01, 2.50000492e-01, 2.22222222e-05])
offset = np.array([-2.50000000e-01,  1.00000290e+00,  2.00000608e+00,  2.00000304e+00, 2.00000608e+00,  1.33333739e+00, -7.18345516e-07, -1.11111111e-01, -1.11111111e-01, -1.96991638e-06, -1.11111111e-01])

x_scaled = x * scale + offset
x_scaled

#%%
from keras.losses import mean_squared_error
for obj in model_ycol:
    model = load_model(obj["model"])
    y_pred = model.predict(x_scaled)[:,0]
    y_true = test_set[:,13 + obj["ycol"]]
    if obj["ycol"] == 2:
        y_true = np.log10(y_true)
    elif obj['ycol'] == 8 or obj['ycol'] == 9:
        b = y_true < 10.0
        y_true = y_true[b]
        y_pred = y_pred[b]

    mse = mean_squared_error(y_true, y_pred)
    ic(obj, mse, np.var(y_true))
