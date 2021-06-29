#%%
from pyexpat import model
import sys,math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from keras.models import load_model
from icecream.icecream import ic
#%%
mpl.rcParams.update(mpl.rcParamsDefault)

#%%
plt.rcParams['font.family'] ='sans-serif'
plt.rcParams["figure.subplot.left"] = 0.2
plt.rcParams["figure.subplot.right"] = 0.95
plt.rcParams["figure.subplot.bottom"] = 0.20
plt.rcParams["figure.subplot.top"] = 0.95
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.major.width'] = 1.0
plt.rcParams['ytick.major.width'] = 1.0
plt.rcParams['axes.labelsize'] = 30
plt.rcParams['font.size'] = 22
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['savefig.dpi'] = 200
plt.rcParams['figure.facecolor'] = 'white'

#%%
model_k = load_model('/Users/murase/work/oacis/public/Result_development/60889d1d709711c3ed44548c/60889d30709711c3ed445495/60889d30709711c3e9b4fbb0/model.h5')
model_kk = load_model('/Users/murase/work/oacis/public/Result_development/60889d1d709711c3ed44548c/60889d30709711c3ed44549b/60889d30709711c3e9b4fbb6/model.h5')
model_cc = load_model('/Users/murase/work/oacis/public/Result_development/60889d1d709711c3ed44548c/60889d30709711c3ed44549d/60889d30709711c3e9b4fbb8/model.h5')
model_o = load_model('/Users/murase/work/oacis/public/Result_development/60889d1d709711c3ed44548c/60889d30709711c3e9b4fbae/60889d30709711c3e9b4fbbc/model.h5')
model_ck = load_model('/Users/murase/work/oacis/public/Result_development/60889d1d709711c3ed44548c/60889d30709711c3e9b4fbac/60889d30709711c3e9b4fbba/model.h5')
model_ow = load_model('/Users/murase/work/oacis/public/Result_development/60889d1d709711c3ed44548c/6088c44f709711c3e9b4fbbe/6088c44f709711c3e9b4fbc0/model.h5')

#%%
# 9 variables
bounds = [(0.0,1.0)]*9

#%%
def evaluate(x):
    target_k = (10.0, 3.0)
    target_kk = (0.1, 0.02)
    target_cc = (0.05, 0.01)
    target_o = (0.05, 0.01)
    d = 0.0
    d += (model_k.predict([list(x)])[0][0] - target_k[0])** 2 / target_k[1]
    d += (model_kk.predict([list(x)])[0][0] - target_kk[0])** 2 / target_kk[1]
    d += (model_cc.predict([list(x)])[0][0] - target_cc[0])** 2 / target_cc[1]
    d += (model_o.predict([list(x)])[0][0] - target_o[0])** 2 / target_o[1]
    return d

#%%
evaluate(np.array([0.0]*9))

#%%
evaluate([1.00000000e+00, 1.12306991e-03, 2.40381195e-03, 1.34050805e-02,
       1.00000000e+00, 5.88440616e-04, 1.00000000e+00, 5.92353677e-03,
       0.00000000e+00])

#%%
from scipy import optimize

results = dict()
results['shgo'] = optimize.shgo(evaluate, bounds)
results['shgo']

#%%
results['DA'] = optimize.dual_annealing(evaluate, bounds)
results['DA']

#%%
results['BH'] = optimize.basinhopping(evaluate, bounds)
results['BH']

#%%
results['shgo_sobol'] = optimize.shgo(evaluate, bounds, n=200, iters=5, sampling_method='sobol')
results['shgo_sobol']

# %%
# results['DA'] = optimize.dual_annealing(eggholder, bounds)
# results['DA']
# # %%
# results['DE'] = optimize.differential_evolution(eggholder, bounds)
# results['DE']
# # %%
# results['BH'] = optimize.basinhopping(eggholder, bounds)
# results['BH']
# # %%
# results['shgo_sobol'] = optimize.shgo(eggholder, bounds, n=200, iters=5,
#                                       sampling_method='sobol')
# results['shgo_sobol']
# # %%
# fig = plt.figure()
# ax = fig.add_subplot(111)
# im = ax.imshow(eggholder(xy), interpolation='bilinear', origin='lower',
#                cmap='gray')
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# 
# def plot_point(res, marker='o', color=None):
#     ax.plot(512+res.x[0], 512+res.x[1], marker=marker, color=color, ms=10)
# 
# plot_point(results['BH'], color='y')  # basinhopping           - yellow
# plot_point(results['DE'], color='c')  # differential_evolution - cyan
# plot_point(results['DA'], color='w')  # dual_annealing.        - white
# 
# # SHGO produces multiple minima, plot them all (with a smaller marker size)
# plot_point(results['shgo'], color='r', marker='+')
# plot_point(results['shgo_sobol'], color='r', marker='x')
# for i in range(results['shgo_sobol'].xl.shape[0]):
#     ax.plot(512 + results['shgo_sobol'].xl[i, 0],
#             512 + results['shgo_sobol'].xl[i, 1],
#             'ro', ms=2)
# 
# ax.set_xlim([-4, 514*2])
# ax.set_ylim([-4, 514*2])
# plt.show()
# %%
