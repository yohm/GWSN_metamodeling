#%%
import sys,math
from pprint import pprint as pp
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from icecream import ic

#%%
mpl.rcParams.update(mpl.rcParamsDefault)

# %%
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
plt.rcParams['xtick.major.pad'] = 8
plt.rcParams['ytick.major.pad'] = 8

#%%
model_k = load_model('results/model_k.h5')
model_kk = load_model('results/model_kk.h5')
model_w = load_model('results/model_w.h5')
model_cc = load_model('results/model_cc.h5')
model_o = load_model('results/model_o.h5')
model_ck = load_model('results/model_ck.h5')
model_ow = load_model('results/model_ow.h5')
model_perc_a = load_model('results/model_perc_a.h5')
model_perc_d = load_model('results/model_perc_d.h5')

#%%
scale = [2.50000000e-04, 3.33335264e-01, 5.00002063e-01, 5.00001303e-01, 5.00001520e-01, 3.33335794e-01, 5.00000359e-01, 1.11111111e-01, 1.11111111e-01, 2.50000492e-01, 2.22222222e-05]
offset = [-2.50000000e-01,  1.00000290e+00,  2.00000608e+00,  2.00000304e+00, 2.00000608e+00,  1.33333739e+00, -7.18345516e-07, -1.11111111e-01, -1.11111111e-01, -1.96991638e-06, -1.11111111e-01]

#%%
base_parameters = { "net_size":2000,"p_tri":0.001,"p_r":0.001,"p_nd":0.001,
                    "p_ld":0.01,"aging":0.003,"w_th":0.5,"w_r":1.0,
                    "q":2,"F":2,"alpha":0.0,"t_max":50000}

#%%
def calc_p_tri_dependency(base_param, model):
    predicted = []
    params = np.array([[
        base_param['net_size'],
        math.log10(base_param['p_tri']),
        math.log10(base_param['p_r']),
        math.log10(base_param['p_nd']),
        math.log10(base_param['p_ld']),
        math.log10(base_param['aging']),
        base_param['w_r'],
        base_param['q'],
        base_param['F'],
        base_param['alpha'],
        base_param['t_max']
        ]])
    p_tris = np.arange(-3.0,-0.1,0.01)
    for p_tri in p_tris:
        params[0][1] = p_tri
        scaled_params = params*np.array(scale)+np.array(offset)
        predicted.append( model.predict(scaled_params)[0][0] )
    return 10**p_tris,np.array(predicted)

#%%
import os,sys
sys.path.append( os.environ['OACIS_ROOT'] )
import oacis

#%%
def get_simulation_data(base_params, x_key = 'p_tri', y_key = 'average_degree'):
    sim = oacis.Simulator.find_by_name("wsn_all")
    base_ps = sim.find_or_create_parameter_set( base_params )
    ps_list = list(base_ps.parameter_sets_with_different( x_key ))
    xs_sim = [ps.v()[x_key] for ps in ps_list]
    ys_sim = [ps.average_result(y_key)[0] for ps in ps_list]
    return xs_sim,ys_sim


#%%
def plot_p_tri_dep(base_params, model, y_key, y_label, ylim = None, ylog = False):
    xs_pred,ys_pred = calc_p_tri_dependency(base_params, model)
    xs_sim, ys_sim = get_simulation_data(base_params, y_key=y_key)
    ic(xs_sim, ys_sim)
    plt.clf()
    plt.xscale('log')
    plt.xlabel(r'$p_{\Delta}$')
    plt.ylabel(y_label)
    plt.xlim([1e-3,1e0])
    if ylim:
        plt.ylim(ylim)
    if ylog:
        ys_pred = 10.0 ** ys_pred
        ic(ys_pred)
        plt.yscale('log')
    plt.plot(xs_pred,ys_pred, '--', color='royalblue')
    plt.plot(xs_sim, ys_sim, 'o', color='blue')

#%%
plot_p_tri_dep(base_parameters, model_k, 'average_degree', r'$\langle k \rangle$')
#plt.show()
plt.savefig('k_p_tri.pdf')

#%%
plot_p_tri_dep(base_parameters, model_w, 'average_link_weight', r'$\langle w \rangle$', ylim=[1e0,150], ylog=True)
plt.savefig('w_p_tri.pdf')

#%%
plot_p_tri_dep(base_parameters, model_kk, 'pcc_k_knn', r'$\rho_{k}$', [-0.2,0.2])
plt.savefig('rho_k_p_tri.pdf')

#%%
plot_p_tri_dep(base_parameters, model_cc, 'clustering_coefficient', r'$C$')
plt.savefig('cc_p_tri.pdf')

#%%
plot_p_tri_dep(base_parameters, model_o, 'link_overlap', r'$O$')
plt.savefig('o_p_tri.pdf')

# %%
plot_p_tri_dep(base_parameters, model_ck, 'pcc_c_k', r'$\rho_{ck}$', [-0.8,0.2])
plt.savefig('rho_ck_p_tri.pdf')
# %%
plot_p_tri_dep(base_parameters, model_ow, 'pcc_link_overlap_weight', r'$\rho_{ow}$', [0,0.6])
plt.savefig('rho_ow_p_tri.pdf')
#plt.show()
# %%
plot_p_tri_dep(base_parameters, model_perc_a, 'percolation_fc_ascending', r'$(1-f_c^a)\langle k \rangle$', [0.0,5.0])
plt.savefig('fca_p_tri.pdf')
#plt.show()

# %%
plot_p_tri_dep(base_parameters, model_perc_d, 'percolation_fc_descending', r'$(1-f_c^d)\langle k \rangle$', [0.0,5.0])
plt.savefig('fcd_p_tri.pdf')
#plt.show()

# %%
