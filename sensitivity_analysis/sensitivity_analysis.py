#%%
import sys,math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from keras.models import load_model
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
from SALib.sample import saltelli
from SALib.analyze import sobol

#    [p_tri, p_r, p_nd, p_ld, aging, w_r, q, F,alpha]
num_vars = 11
problem = {
    'num_vars': num_vars,
    'names': [r'$N$', r'$p_{\Delta}$', r'$p_r$', r'$p_{nd}$', r'$p_{ld}$', r'$A$', r'$w_r$', r'$q$', r'$F$' , r'$\alpha$', r'$t$'],
    'bounds': [ [0.0,1.0] for i in range(num_vars) ]
}
ic(problem)

#%%
# order : N F q alpha p_r p_delta w_r p_nd p_ld A t
display_order = [0, 5, 4, 7, 8, 9, 6, 2, 1, 3, 10]


#%%
def calc_sensitivity_index(problem, model):
    param_values = saltelli.sample(problem, 16384, skip_values=32768, calc_second_order=False)
    y = model.predict(param_values)
    si = sobol.analyze(problem, y.flatten(), print_to_console=True, calc_second_order=False)
    return si

#%%
def plot_s1_st(problem,si,legend=True):
    plt.clf()
    # x = np.arange( problem['num_vars'] )[::-1]
    x = np.array(display_order)
    ic(x)
    width = 0.4  # the width of the bars
    plt.xticks(x, problem['names'])
    plt.ylabel(r'$S_1,S_T$')
    plt.bar(x-width/2, si['S1'], width=width, yerr=si['S1_conf'], label=r'$S_1$', error_kw=dict(lw=1, capsize=3, capthick=1))
    plt.bar(x+width/2, si['ST'], width=width, yerr=si['ST_conf'], label=r'$S_T$', error_kw=dict(lw=1, capsize=3, capthick=1))
    if legend:
        plt.legend()
    plt.ylim(bottom=0.0)



#%%
si = calc_sensitivity_index(problem, model_k)
plot_s1_st(problem, si)
plt.savefig('SA_k.pdf')
#plt.show()

#%%
si = calc_sensitivity_index(problem, model_w)
plot_s1_st(problem, si, legend=False)
plt.savefig('SA_w.pdf')

#%%
si = calc_sensitivity_index(problem, model_kk)
plot_s1_st(problem, si, legend=False)
plt.savefig('SA_kk.pdf')

# %%
si = calc_sensitivity_index(problem, model_cc)
plot_s1_st(problem, si, legend=False)
plt.savefig('SA_cc.pdf')

# %%
si = calc_sensitivity_index(problem, model_o)
plot_s1_st(problem, si, legend=False)
plt.savefig('SA_o.pdf')
# %%
si = calc_sensitivity_index(problem, model_ck)
plot_s1_st(problem, si, legend=False)
plt.savefig('SA_ck.pdf')
# %%
si = calc_sensitivity_index(problem, model_ow)
plot_s1_st(problem, si, legend=False)
plt.savefig('SA_ow.pdf')

#%%
si = calc_sensitivity_index(problem, model_perc_a)
plot_s1_st(problem, si, legend=False)
plt.savefig('SA_perc_a.pdf')
#plt.show()
# %%
si = calc_sensitivity_index(problem, model_perc_d)
plot_s1_st(problem, si, legend=False)
plt.savefig('SA_perc_d.pdf')


# %%
