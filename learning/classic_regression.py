#%%
import sys,math,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from icecream import ic

#%%
if len(sys.argv) == 3:
    filename = sys.argv[1]
    input_json = sys.argv[2]
else:
    filename = "sample_training_data.txt"
    input_json = "classic_reg_input.json"
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
#         "polynomial_degree": 1,
#         "alpha": 0.1,
#         "l1_ratio": 0.7,
#         }
def load_input_json(json_path):
    with open(json_path) as f:
        return json.load(f)
h_params = load_input_json(input_json)
ic(h_params)

#%%
# parameters:
x = np.log10(training[:,1:6])
x = np.concatenate([training[:,0:1],x,training[:,7:12]], axis=1 )
ic(np.shape(x))

#%%
def scale_input(data):
    #from sklearn.preprocessing import StandardScaler
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    #scaler = StandardScaler()
    scaled = scaler.fit_transform(data)
    return (scaler, scaled)

scaler, x_all = scale_input(x)
ic(scaler.min_, scaler.scale_)

#%%
y_all = training[:,13 + h_params['ycol']]
if h_params['ycol'] == 2:
    y_all = np.log10(y_all)  # take logarithm for link weight
ic(y_all.shape)

#%%
# Split data in train set and test set
n_samples = x_all.shape[0]
x_train, y_train = x_all[:n_samples // 2], y_all[:n_samples // 2]
x_test, y_test = x_all[n_samples // 2:], y_all[n_samples // 2:]

#%%
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

model = Pipeline([('poly', PolynomialFeatures(degree=3)),
                  ('linear', ElasticNet(alpha=h_params['alpha'], l1_ratio=h_params['l1_ratio'], fit_intercept=False))])
#enet = ElasticNet(alpha=h_params['alpha'], l1_ratio=h_params['l1_ratio'])

y_pred_enet = model.fit(x_train, y_train).predict(x_test)
#r2_score_enet = r2_score(y_test, y_pred_enet)
mse = mean_squared_error(y_test, y_pred_enet)

ic(y_pred_enet[-10:], y_test[-10:], np.sqrt(mse))

# %%
print( json.dumps({"mse": mse}) )
# %%
