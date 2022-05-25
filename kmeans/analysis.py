import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
ell_err_spir=pd.read_csv(r'E:\二分类论文\预测结果\重复\椭圆分错为涡旋\椭圆分错为涡旋.csv',header=0)
ell_err_spir_H_EW=ell_err_spir.values[:,11]
ell_err_spir_H_flux=ell_err_spir.values[:,12]

ell_corr_ell=pd.read_csv(r'E:\二分类论文\预测结果\重复\椭圆分为椭圆\ell_corr_ell.csv',header=0)
ell_corr_ell_H_EW=ell_corr_ell.values[:,11]
ell_corr_ell_H_flux=ell_corr_ell.values[:,12]


spir_err_ell=pd.read_csv(r'E:\二分类论文\预测结果\重复\涡旋分错为椭圆\涡旋分错为椭圆.csv',header=0)
spir_err_ell_H_EW=spir_err_ell.values[:,11]
spir_err_ell_H_flux=spir_err_ell.values[:,12]

spir_corr_spir=pd.read_csv(r'E:\二分类论文\预测结果\重复\涡旋分为涡旋\spir_corr_spir.csv',header=0)
spir_corr_spir_H_EW=spir_corr_spir.values[:,11]
spir_corr_spir_H_flux=spir_corr_spir.values[:,12]

print('__________________________________________________________')
bi=list(np.arange(0,3,0.1))

# sns.set(font_scale=1)

# plt.subplot(221)
a=sns.distplot(ell_corr_ell_H_EW,bins=bi,kde_kws={"color":"royalblue", "lw":2}, hist_kws={ "color": "royalblue" })  #
b=sns.distplot(ell_err_spir_H_EW,bins=bi,kde_kws={"color":"orange", "lw":2}, hist_kws={ "color": "orange",'alpha':2/3  })  #
c=sns.distplot(spir_err_ell_H_EW,bins=bi,kde_kws={"color":"limegreen", "lw":2}, hist_kws={ "color": "limegreen",'alpha':2/3  })  #

t=sns.distplot(spir_corr_spir_H_EW,bins=bi,kde_kws={"color":"r", "lw":2 }, hist_kws={ "color": "r",'alpha':2/3  })
plt.legend(labels=['ell_corr_ell_H_EW','ell_err_spir_H_EW','spir_err_ell_H_EW','spir_corr_spir_H_EW'])
plt.xlim([0.5,3])
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
a.set_xlabel('Hα_EW',fontsize=20)
a.set_ylabel('Density',fontsize=20)
#
#
# plt.subplot(222)
# t=sns.distplot(s_H_EW,bins=bi,kde_kws={"color":"seagreen", "lw":2 }, hist_kws={ "color": "r",'alpha':2/3  })
# plt.legend(labels=['s_H_EW'])
# plt.xlim([0,3])
# t.set_xlabel('(b)',fontsize=20)
# t.set_ylabel('')
#
#
# plt.subplot(223)
# c=sns.distplot(e_H_flux,bins=20,kde_kws={"color":"k", "lw":2 }, hist_kws={ "color": "b" ,'alpha':2/3 })
# plt.legend(labels=['III'])
# # plt.xlim([0,0.16])
# c.set_xlabel('(c)',fontsize=20)
# c.set_ylabel('')
#
#
# plt.subplot(224)
# d=sns.distplot(s_H_flux,bins=20,kde_kws={"color":"b", "lw":2 }, hist_kws={ "color": "r" ,'alpha':2/3 })
# plt.legend(labels=['IV'])
# # plt.xlim([0,0.16])
# d.set_xlabel('(d)',fontsize=20)
# d.set_ylabel('')
plt.show()