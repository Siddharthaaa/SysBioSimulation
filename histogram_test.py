# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 11:14:46 2019

@author: imb30
"""

from bioch_sim import *

from sklearn.cluster import KMeans, MeanShift, k_means

TEST = True

sigma = 0.1

lage = 0.1

scale = 0.1

s1 = np.random.normal(lage + scale * 1, sigma, 500)

s2 = np.random.normal(lage + scale * 5, sigma, 300)
s3 = np.random.normal(lage + scale * 9, sigma, 600)

#s1 = np.random.uniform(size =100000)

s=np.hstack((s1,s2,s3))

ax = plot_hist(s,  max_range=1, exp_maxi=2)

kde = gaussian_kde(s2, bw_method = "scott")
kde.integrate_box_1d(-10000,10000)
#ax.set_title("Multimodal")

print(get_multimodal_scores(s)[0])


if TEST:
    kde_test = []
  
    dists = []
    for i in range(10):
        sigma = 0.1

        lage = 0.1
        scale = 0.1
        s1 = np.random.normal(lage + scale * 2, sigma, int(np.random.uniform() * 1000))    
#        s2 = np.random.normal(lage + scale * 5, sigma, int(np.random.uniform() * 0))
        s2 = np.random.uniform(0.1,0.9,size = int((np.random.uniform() * 400)))
        s3 = np.random.normal(lage + scale * 8, sigma, int(np.random.uniform() * 1000))
        
        #s1 = np.random.uniform(size =100000)
        s=np.hstack((s1,s2,s3))
        dists.append(s)
#        plot_hist(s, exp_maxi=2)
        kde_test.append(get_multimodal_scores(s,1)[0][:,0].max())
    
    
    indices = np.argsort(kde_test)
    
    fig, axs = plt.subplots(2,5)
    axs=axs.flatten()
    i=0
    for ax in axs:
            plot_hist(dists[indices[i]],"",ax= ax, max_range=0.8)
#            ax.axis("off")
            ax.set_yticklabels([])
            i+=1