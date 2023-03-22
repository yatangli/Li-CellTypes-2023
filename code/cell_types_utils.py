#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 11:40:35 2023

@author: yatangli
"""

import numpy as np
import math

from matplotlib import pyplot as plt
from matplotlib import colors
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import SpectralClustering
from sklearn import cluster

import seaborn as sns
import scipy
from scipy.spatial import distance
import hdf5storage
import pandas as pd
import timeit
from sklearn.mixture import GaussianMixture
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import FuncFormatter
import sklearn
from sklearn.decomposition import SparsePCA
import bisect
from scipy.spatial.distance import pdist, squareform
from sklearn import datasets
# from fastcluster import linkage

#%% define global variables      
fig_width = 8
fig_height = 18
#%%
def load_data_new(folder_path, data_path):
    
    neuron = hdf5storage.loadmat(folder_path+data_path['neuron'])
    chirp = hdf5storage.loadmat(folder_path+data_path['chirp'])
    dl_lr = hdf5storage.loadmat(folder_path+data_path['dl_lr'])
    mb = hdf5storage.loadmat(folder_path+data_path['mb'])
    color = hdf5storage.loadmat(folder_path+data_path['color'])
    st = hdf5storage.loadmat(folder_path+data_path['st'])
    rf = hdf5storage.loadmat(folder_path+data_path['rf'])
    
    return neuron,chirp,dl_lr,mb,color,st,rf
# %%
def adjacent_values(vals, q1, q3):
    upper_adjacent_value = q3 + (q3 - q1) * 1.5
    upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

    lower_adjacent_value = q1 - (q3 - q1) * 1.5
    lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
    return lower_adjacent_value, upper_adjacent_value

def violin_plot(x,ax,pos=1,facecolor='orange',median_color='white',mean_color='blue'):
    parts = ax.violinplot(
        x, [pos], showmeans=False, showmedians=False, vert=False,
        showextrema=False, points=200)
    for pc in parts['bodies']:
        pc.set_facecolor(facecolor)
        pc.set_edgecolor('black')
        pc.set_alpha(1)
        
    quartile1, medians, quartile3 = np.percentile(x, [25, 50, 75], axis=0)
    whiskers = np.array([adjacent_values(sorted(x), quartile1, quartile3)])
    whiskers_min, whiskers_max = whiskers[:, 0], whiskers[:, 1]
    inds = np.arange(pos, pos+1)
    means = np.mean(x,axis=0)
    ax.scatter(medians, inds, marker='o', color=median_color, s=30, zorder=3)
    ax.scatter(means, inds, marker='|', color=mean_color, s=100, zorder=3)
    ax.vlines(means,inds[0]-0.1,inds[0]+0.1,color='blue', linestyle='-',lw=3,zorder=3)
    ax.hlines(inds, quartile1, quartile3, color='black', linestyle='-', lw=5)
    ax.hlines(inds, whiskers_min, whiskers_max, color='black', linestyle='-', lw=1)
#%% define my on PCA
def pca_x(X,centralize=True):
    """
    SVD approach for principal components analysis
    PV: principal axis 
    PC: principal components
    lambda: variance explained"""
    [n, p] = np.shape(X)                
    X_0 = np.zeros((n,p))
    if centralize:
        for i in range(p):
            X_0[:,i] =  X[:,i]-np.mean(X[:,i])
    else:
        X_0 = X
    [U,S,V] = np.linalg.svd(X_0)
    V = V.T
    #u @ np.diag(s) @ vh = (u * s) @ vh 
    # X = U*S*V'. C=X_centered'*X_centered/(n-1)=V*S*U'U*S*V'/(n-1)=V*S^2/(n-1)*V'
    # Columns of V are principal directions/axes.
    # Columns of US or XV are principal components ("scores").   
     
    #  the following is covraince matrix approach; these two give the same result
    #  X_cov = X_0.T@X_0 #covariance matrix
    # eigenvalues = np.linalg.eigvals(X_cov)
    # eigenvalues = eigenvalues/np.sum(eigenvalues)  
               
    lamda = np.square(S)/(n-1)
    lamda = lamda/np.sum(lamda)
    PV = V
    PC = X_0@PV #principal components
    return PV, lamda, PC  
#%% define a plot function
def plot_cluster(X,labels,file_save=[], fig_zoom=1, sampling_rate=5,plot_bool=False):
    [n_cells, n_samples] = np.shape(X)
    xtick_inter = int(max(5,np.floor(n_samples/(sampling_rate*5*3))*5)) #second; set for 3 ticks
    xticks = range(0,n_samples,sampling_rate*xtick_inter)
    xtick_num = len(xticks)
    xticklabels = []
    for i in range(xtick_num):
        xticklabels.append(str(xticks[i]/sampling_rate))
    X_sorted = np.zeros((n_cells, n_samples))
    labels_unique = np.unique(labels)
    n_cluster = len(labels_unique)
    X_cluster = np.zeros((n_cluster,n_samples))
    X_cluster_error = np.zeros((n_cluster,n_samples))
    n_cells_per_cluster = np.zeros(n_cluster,dtype=np.int64)
    # re-assign labels so that 0 is the largest cluster 
    labels_sorted = np.ones_like(labels)*1e3
    labels_count = np.zeros_like(labels_unique)
    for i in range(n_cluster):
        labels_count[i] = sum(np.isin(labels,labels_unique[i]))
        
    # # sort the cluster so that the 1st cluster has the most neurons
    # # this is not necessary, keep orignial order @ 20200614
    # labels_unique_sorted = np.argsort(labels_count)[::-1]
    # for i in range(n_cluster):
    #     labels_sorted[labels==labels_unique_sorted[i]] = i

    labels_sorted = labels  
    
    k = 0
    for i in range(n_cluster):
        n_cells_per_cluster[i] = int(sum(labels_sorted==i))
        X_sorted[k:k+n_cells_per_cluster[i],:] = X[labels_sorted==i,:]
        X_cluster[i,:] = np.nanmean(X[labels_sorted==i,:],axis=0)
        X_cluster_error[i,:] = np.nanstd(X[labels_sorted==i,:],axis=0)
        k = k+n_cells_per_cluster[i] 
    if plot_bool:
        plt.figure(figsize=(fig_width*fig_zoom,np.min([n_cells/16,fig_height])*fig_zoom))
        axes= plt.axes()
        vmin = np.min(X_sorted)
        vmax = np.max(X_sorted)
        if vmin<0 and np.abs(vmin)>np.abs(vmax)/2:
            norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)      
            plt.pcolormesh(X_sorted, cmap='bwr', norm=norm)   
        else:
            plt.imshow(X_sorted, aspect='auto', origin='lower', cmap='viridis')#'viridis' plasma' 'cividis' cmap='bwr'
        k = 0
        for i in range(n_cluster):
            k = k+n_cells_per_cluster[i]
            plt.plot([0,n_samples-0.9],[k-0.5,k-0.5],'r-')
        axes.set_xticks(xticks)
        axes.set_xticklabels(xticklabels)
        #plt.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis');
        plt.colorbar(aspect=60)
        if len(file_save)>0:
            plt.savefig(file_save+'.png',bbox_inches='tight')
            plt.savefig(file_save+'.svg',bbox_inches='tight')
            plt.savefig(file_save+'.pdf',bbox_inches='tight')
        
        
        plt.figure(figsize=(fig_width,n_cluster/n_samples*fig_width*2))
        axes= plt.axes()
        plt.imshow(X_cluster, aspect='auto', origin='lower',cmap='viridis')#'viridis' plasma' 'cividis'
        axes.set_xticks(xticks)
        axes.set_xticklabels(xticklabels)
        # axes.tick_params(labelsize=25)
        # plt.colorbar().ax.tick_params(labelsize=25)
        plt.colorbar()
        plt.show()
        
    
    
    return X_cluster, X_cluster_error, X_sorted, labels_count
#%% define a plot function
def imshow_clusters(X_cluster,sampling_rate=5):
    [n_cluster, n_samples] = np.shape(X_cluster)
    xtick_inter = int(max(5,np.floor(n_samples/(sampling_rate*5*3))*5)) #second; set for 3 ticks
    xticks = range(0,n_samples,sampling_rate*xtick_inter)
    xtick_num = len(xticks)
    xticklabels = []
    for i in range(xtick_num):
        xticklabels.append(str(xticks[i]/sampling_rate))
           
    plt.figure(figsize=(3*fig_width,3*np.min([n_cluster/4,fig_height])))
    axes= plt.axes()
    plt.imshow(X_cluster,aspect='auto',cmap='viridis')#'viridis' plasma' 'cividis'
    axes.set_xticks(xticks)
    axes.set_xticklabels(xticklabels)
    axes.tick_params(labelsize=25)
    # plt.rcParams.update({'font.size': 25})
    plt.colorbar().ax.tick_params(labelsize=25)
    plt.show()
       
#%% show image
def imshow_X(X,title=None,sampling_rate=5,cmap='viridis',aspect='auto'):
    if len(X.shape)==1:
        X = np.expand_dims(X, axis=1)
    [n_cells, n_samples] = np.shape(X)
    # if sampling_rate == 1:
    #     plt.figure(figsize=(fig_width*2,np.min([n_cells/16,fig_height*2])))
    # else:
    #     plt.figure(figsize=(fig_width,np.min([n_cells/16,fig_height])))
    plt.figure(figsize=(fig_width,np.min([n_cells/16,fig_height])))
    axes= plt.axes()
    plt.imshow(X,aspect=aspect,cmap=cmap)#'viridis' plasma' 'cividis'
    # print(sampling_rate)
    xtick_inter = int(max(5,np.floor(n_samples/(sampling_rate*5*3))*5)) #second; set for 3 ticks or 5 seconds
    xticks = range(0,n_samples,sampling_rate*xtick_inter)
    xtick_num = len(xticks)
    if xtick_num<3:
        xtick_inter = 1 #sec
        xticks = range(0,n_samples,sampling_rate*xtick_inter)
        xtick_num = len(xticks)
        
    axes.set_xticks(xticks)
    xticklabels = []
    for i in range(xtick_num):
        xticklabels.append(str(xticks[i]/sampling_rate))
    axes.set_xticklabels(xticklabels)
    plt.title(title)
    plt.colorbar()
    plt.show()

#%% show image
def imshow_PV(X,title=None,sampling_rate=5,cmap='viridis',ytick_step=2,vmin=0,vmax=1,origin='lower'):
    if len(X.shape)==1:
        X = np.expand_dims(X, axis=1)
    [n_pv, n_sample] = np.shape(X)
    # if sampling_rate == 1:
    #     plt.figure(figsize=(fig_width*2,np.min([n_cells/16,fig_height*2])))
    # else:
    #     plt.figure(figsize=(fig_width,np.min([n_cells/16,fig_height])))
    fig_unit_size = 0.2
    plt.figure(figsize=(fig_unit_size*n_sample,fig_unit_size*n_pv))
    axes= plt.axes()
    plt.imshow(X,aspect='equal',cmap=cmap,vmin=vmin, vmax=vmax,origin=origin)#'viridis' plasma' 'cividis'
    # print(sampling_rate)
    axes.set_xticks([])     
    yticks = np.asarray(range(0,n_pv,ytick_step))
    ytick_num = len(yticks)        
    axes.set_yticks(yticks)
    yticklabels = []
    for i in range(ytick_num):
        yticklabels.append(str(yticks[i]))
    axes.set_yticklabels(yticklabels)
    plt.title(title)
    v = np.linspace(vmin, vmax, 3, endpoint=True)
    plt.colorbar(ticks=v)
    # plt.show()    
#%% normalize responses at each row to [0,1]
def norm_x(x,formula = 0):
    y = np.copy(x)
    if x.ndim == 2:
        for i,row in enumerate(y):
            if formula == 0:
                temp = row - min(row)
                y[i] = temp/max(temp)
            else:
                y[i] = row/max([abs(max(row)),abs(min(row))])  
    elif x.ndim == 1:
        if formula == 0:
            temp = x-min(x)
            y = temp/max(temp)
    return y
#%% standarize each columns / features
#%%determine the optimal cluster by fitting Gaussian mixture model to data
def optimal_cluster_gmm(X,n_components_max,random_init=False,n_init=100):
    # n_components_max = 50
    n_neuron = X.shape[0]
    range_n_clusters = list(range(2,n_components_max+1))
    count_n_clusters = len(range_n_clusters)
    # print(random_init)
    if random_init:
        bic = np.ones(count_n_clusters)*np.nan
        aic = np.ones(count_n_clusters)*np.nan
        silhouette_score = np.ones(count_n_clusters)*np.nan
        labels_arr = np.ones((n_neuron,count_n_clusters),dtype=np.dtype('uint8'))      
    else:
        bic = np.ones((count_n_clusters,n_init))*np.nan
        aic = np.ones((count_n_clusters,n_init))*np.nan
        silhouette_score = np.ones((count_n_clusters,n_init))*np.nan
        labels_arr = np.ones((n_neuron,count_n_clusters,n_init),dtype=np.dtype('uint8'))
        
    # print(bic.shape)
    for i,n_cluster in enumerate(range_n_clusters):
        print(n_cluster)
        if random_init:
            gmm = GaussianMixture(n_components=n_cluster,covariance_type='diag', n_init = n_init, max_iter=1000)
            gmm.fit(X)
            labels = gmm.predict(X)
            labels_arr[:,i] = labels
            bic[i] = gmm.bic(X)
            aic[i] = gmm.aic(X)
            silhouette_score[i] = sklearn.metrics.silhouette_score(X,labels) 
        else:
            for j in range(n_init):
                # print(j)
                gmm = GaussianMixture(n_components=n_cluster,covariance_type='diag', random_state = j, max_iter=1000)
                gmm.fit(X)
                labels = gmm.predict(X)
                labels_arr[:,i,j] = labels
                bic[i,j] = gmm.bic(X)
                aic[i,j] = gmm.aic(X)
                silhouette_score[i,j] = sklearn.metrics.silhouette_score(X,labels) 
                

        # # calculate bic, it's the same as gmm.bic(x), verified.
        # l = gmm.score(pc_k)*n
        # m = (k*2+1)*n_cluster_optimal - 1
        # bic_cal = -2*l + m*np.log(n)
    if random_init:
        bic_plot = bic
        aic_plot = aic
        silhouette_score_plot = silhouette_score
    else:
        bic_plot = np.min(bic,axis=1)
        aic_plot = np.min(aic,axis=1)
        silhouette_score_plot = np.min(silhouette_score,axis=1)
        
    
    n_cluster = np.argmin(bic_plot)+2
    bic_idx = np.argmin(bic,axis=1)
    labels_arr_optimal = labels_arr[:,n_cluster-2,bic_idx[n_cluster-2]]

    
        
    
    
    # bic_delta = bic[1:] - bic[:-1]
    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(fig_width,fig_height*0.68))
    axs[0].plot(range_n_clusters,bic_plot,marker='+')
    axs[0].yaxis.set_major_formatter(FormatStrFormatter('%.2e'))
    # xticks_cluster = range(1,n_components_max+1,5)
    # axs[0].set_xticks(xticks_cluster)
    axs[0].legend(['BIC scores'])
    
    
    axs[1].plot(range_n_clusters,aic_plot,marker='+')
    axs[1].yaxis.set_major_formatter(FormatStrFormatter('%.2e'))
    # xticks_cluster = range(1,n_components_max+1,5)
    # axs[0].set_xticks(xticks_cluster)
    axs[1].legend(['AIC scores'])
    
    
    # axs[1].plot(range_n_clusters[1:],bic_delta,marker='+')  
    # axs[1].yaxis.set_major_formatter(FormatStrFormatter('%.2e'))
    # # axs[1].set_xticks(xticks_cluster)
    # axs[1].legend(['Gradient of BIC scores'])
    
    axs[2].plot(range_n_clusters,silhouette_score_plot,marker='+')  
    axs[2].yaxis.set_major_formatter(FormatStrFormatter('%.2e'))
    axs[2].legend(['Silhouette scores'])
    
    
    # n_cluster = np.where(bic_delta>0)[0][0]+2
    axs[0].title.set_text('The optimal cluster number is '+ str(n_cluster))
    plt.show() 
    # if n_cluster > 2:
      
    print('The optimal cluster number is '+ str(n_cluster))
    # else:
    #     n_cluster = 6
        # print('Set the optimal cluster number to 6')
        
    return n_cluster, aic, bic, silhouette_score, labels_arr, labels_arr_optimal

def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)
    
    
def plot_cluster_dist(cluster_centers_0, cluster_centers_1=None, plot_bool=False):
    if cluster_centers_1 is None:
        cluster_centers_1 = cluster_centers_0
    n_cluster_0 = cluster_centers_0.shape[0]
    n_cluster_1 = cluster_centers_1.shape[0]
    cluster_dist = np.ones((n_cluster_0,n_cluster_1))*np.nan
    for i in range(n_cluster_0):
        for j in range(n_cluster_1):
            cluster_dist[i,j] = distance.euclidean(cluster_centers_0[i,:], cluster_centers_1[j,:])   
    if plot_bool:
        plt.figure(figsize=(fig_width*0.8,fig_width*0.8))
        plt.imshow(cluster_dist,aspect='equal',cmap='viridis', interpolation="none", extent=[1,n_cluster_0,n_cluster_1,1])#'viridis' plasma' 'cividis'
        plt.colorbar()
        plt.show()
    return cluster_dist
def pca_analysis(X, variance_explained,plot_flag=True):
    if plot_flag:
        imshow_X(X,'original X')
    n_cells, n_samples = np.shape(X)
    pv, lamda, pc = pca_x(X,centralize=True)
    # show PCA results
    # plt.figure(figsize=(fig_width,fig_height/5))
    # plt.plot(np.log(lamda),marker='o')
    # plt.savefig('G:/analysis/functional_map/test.svg')
    # choos PCs that explain >80% of the variance
    lamda_cumsum = np.cumsum(lamda)
    if plot_flag:
        fig, axes = plt.subplots(2, 1, sharex=True, figsize=(fig_width,fig_height/3))
        axes[0].plot(lamda[:100],marker='o')
        axes[1].plot(lamda_cumsum[:100],marker='o')
        axes[1].set_ylabel('variance explained')
        axes[1].set_xlabel('number of PCs')

    k = np.where(lamda_cumsum>variance_explained)[0][0]+1
    pc_k = pc[:,:k]
    pv_k = pv[:,:k]
  
    lamda_array = np.ones([n_samples,1])@np.reshape(lamda,(1,-1))
    pv_scaled = np.multiply(lamda_array,pv)
    
    if plot_flag:
        fig, axes = plt.subplots(2, 1, sharex=True, figsize=(fig_width,np.min([k/4,fig_height])))
        sns.heatmap(pv_scaled[:,:k].T,cmap='viridis',xticklabels=False, yticklabels=False, ax=axes[0])
        axes[0].set_title(str(k)+' principal vectors,'+ str(variance_explained*100) + '% of vraiance explained')
        sns.heatmap(pv_k.T,cmap='viridis',xticklabels=False, yticklabels=False, ax=axes[1])
    
    
        
        # plot principal components 
        imshow_X(pc[:,:k],'principal components',sampling_rate=1)
        # visualize the first 2 PCs
        plt.figure(figsize=(fig_width,fig_height/3))
        plt.scatter(pc[:,0],pc[:,1],s=20,facecolors='none',edgecolors='k',marker='^',alpha=0.3)
        plt.xlabel('PC 1')
        plt.ylabel('PC 2')
    
    
    # reconstruct X with PCs that explain variance_explained of variance
    k =np.where(lamda_cumsum>variance_explained)[0][0]+1
    X_0_new = pc[:,:k]@pv[:,:k].T
    X_new = np.zeros_like(X)
    for i in range(n_samples):
        X_new[:,i] =  X_0_new[:,i]+np.mean(X[:,i])
    if plot_flag:
        imshow_X(X_new,'reconstructed X with '+str(k)+' PCs')
    return k, pc_k, pv_scaled, X_new, lamda_cumsum

def align_to_peak_or_trough(x,sort_idx_input=[],neg_idx=[],plot_bool=False):
    num_neuron,x_len = x.shape
    baseline_ini = np.median(x[:,:3],axis=1)
    baseline_fin = np.median(x[:,-3:],axis=1)
    baseline_fin_idx = baseline_ini > baseline_fin
    baseline = np.copy(baseline_ini)
    baseline[baseline_fin_idx] = baseline_fin[baseline_fin_idx]
    baseline_std_ini = np.std(x[:,:3],axis=1)
    baseline_std_fin = np.std(x[:,-3:],axis=1)
    baseline_std = np.copy(baseline_std_ini)
    baseline_std[baseline_fin_idx] = baseline_std_fin[baseline_fin_idx]
    # define negative responses to moving bars
    peak_amp = np.max(x,axis=1) - baseline
    trough_amp = baseline - np.min(x,axis=1)
    if len(neg_idx) == 0:
        neg_idx = trough_amp > peak_amp
    sort_idx_cal = np.logical_or(peak_amp>3*baseline_std, trough_amp>3*baseline_std)
    if len(sort_idx_input) == 0:
        sort_idx = np.copy(sort_idx_cal)
    else:       
        sort_idx = np.logical_or(sort_idx_input, sort_idx_cal)
    # find peak position for positive response and trough position for negative position
    # print(type(sort_idx))
    # print(sort_idx.shape)
    peak_trough = np.zeros(num_neuron)
    peak_trough[np.logical_not(neg_idx)] = np.argmax(x, axis=1)[np.logical_not(neg_idx)]
    peak_trough[neg_idx] = np.argmin(x, axis=1)[neg_idx]
    
    x_sorted = np.zeros((num_neuron,x_len*2))
    idx_ini_sorted = (x_len - peak_trough).astype(int)
    for i in range(num_neuron):
        if sort_idx[i]:
            x_sorted[i,idx_ini_sorted[i]:idx_ini_sorted[i]+x_len] = x[i,:]
        else:
            x_sorted[i,int(x_len/2):int(x_len/2)+x_len] = x[i,:]
 
    # x_sorted = np.zeros_like(x)
    # idx_ini = np.maximum(0, peak_trough.astype(int)-int(x_len/2)).astype(int)
    # idx_fin = np.minimum(x_len,peak_trough.astype(int)+int(x_len/2)).astype(int)
    # idx_ini_sorted = np.maximum(0,int(x_len/2)-peak_trough.astype(int)+idx_ini).astype(int)
    # idx_fin_sorted = np.minimum(x_len,int(x_len/2)-peak_trough.astype(int)+idx_fin).astype(int)
    # for i in range(num_neuron):
    #     x_sorted[i,idx_ini_sorted[i]:idx_fin_sorted[i]] = x[i,idx_ini[i]:idx_fin[i]]
        
    # compare sorted with unsorted  
    if plot_bool:
        imshow_X(norm_x(x_sorted) ,'sorted')  
        imshow_X(norm_x(x),'raw')
    return x_sorted, sort_idx, neg_idx
def sparse_pca_analysis(X, k_spca=20,plot_bool=False):
    spca = SparsePCA(n_components=k_spca, alpha=1, ridge_alpha=0.01, max_iter=1000, tol=1e-08, method='lars', random_state=0)
    pc_spca = spca.fit_transform(X)
    pv_spca = spca.components_
    idx_neg = np.mean(pv_spca,axis=1)<0
    pv_spca[idx_neg,:] = -pv_spca[idx_neg,:]
    
    # plt.figure(figsize=(fig_width,k_spca*0.2))
    # sns.heatmap(pv_spca,cmap='viridis',xticklabels=False, yticklabels=False)
    [n, p] = np.shape(X)  
    X_0 = np.zeros((n,p))
    for i in range(p):
        X_0[:,i] =  X[:,i]-np.mean(X[:,i])
    pc_spca_pos = X_0@pv_spca.T
    X_0_new_spca = pc_spca_pos@pv_spca
    X_new_spca = np.zeros_like(X)
    for i in range(p):
        X_new_spca[:,i] =  X_0_new_spca[:,i]+np.mean(X[:,i])
    
    # return pc_spca, pv_spca # changed to pc_spca_pos on 20210427 
    idx_sorted_chirp = np.argsort(np.argmax(pv_spca,axis=1))
    pv_spca_sorted = pv_spca[idx_sorted_chirp,:]
    pc_spca_sorted = pc_spca_pos[:,idx_sorted_chirp]
    
    if plot_bool:
        # imshow_X(pc_spca_sorted,'PCs from sparse PCA',sampling_rate=1)
        imshow_X(pv_spca_sorted,'PV from sparse PCA',sampling_rate=1)
    
    return pc_spca_sorted, pv_spca_sorted
def pca_reconstruct(X,pc_spca,pv_spca):
    # n = np.shape(pc_spca)[0]  
    p = np.shape(pv_spca)[1]
    # X_0 = np.zeros((n,p))
    # for i in range(p):
    #     X_0[:,i] =  X[:,i]-np.mean(X[:,i])
    # pc_spca_pos = X_0@pv_spca.T
    X_0_new_spca = pc_spca@pv_spca
    X_new_spca = np.zeros_like(X)
    for i in range(p):
        X_new_spca[:,i] =  X_0_new_spca[:,i]+np.mean(X[:,i])
    # imshow_X(X_0_new_spca,'reconstruced X')
    imshow_X(X_new_spca,'reconstruced X')
    return X_new_spca
def round_up(n, decimals=0): 
    multiplier = 10 ** decimals 
    return math.ceil(n * multiplier) / multiplier

def round_down(n, decimals=0):
    multiplier = 10 ** decimals
    return math.floor(n * multiplier) / multiplier
#%% plot histograms for all genetically labeled clusters
def figure_genetic_hist(df_results, genetic_name, columns_label, columns_name, bins_list,
                        file_save, fac=False, figsize=(16,10), pad=1.0, alpha=1):
    
    fig_row = len(genetic_name)#7 # number of genetic labels
    fig_col = len(columns_label)#6 # number of functional properties
    if not fac:
        fig_col = fig_col+1
    
    # columns_name = ['RF size', 'DSI', 'OSI', 'HI', 'LSI', 'Pref. size', 'SSI']
    fig, axs = plt.subplots(fig_row, fig_col, sharex=False, figsize=figsize)
    fig.tight_layout(pad=pad)
    
    # calculate the relative frequency for histograms
    # for all clusters
    freq_all = []
    for i in range(fig_col-1):
        if i == 0 and columns_label[i]=='rf_size':
            hist, edges = np.histogram(np.log10(df_results[columns_label[i]]), bins_list[i])
        else:
            hist, edges = np.histogram(df_results[columns_label[i]], bins_list[i])
        freq_all.append(hist/float(hist.sum())*100)
    # for each cluster
    freq_clusters = []
    for i in range(fig_row):
        freq_temp = []
        for j in range(fig_col-1):
            if j == 0 and columns_label[j]=='rf_size':
                hist, edges = np.histogram(np.log10(df_results.loc[df_results['genetic_label_num']==i][columns_label[j]]), bins_list[j])
            else:
                hist, edges = np.histogram(df_results.loc[df_results['genetic_label_num']==i][columns_label[j]], bins_list[j])              
            freq_temp.append(hist/float(hist.sum())*100)
        freq_clusters.append(freq_temp)
    
    ylim_list = []
    for i in range(fig_col-1):
        max_temp = np.max(freq_all[i])
        for j in range(fig_row):
            if max_temp < np.max(freq_clusters[j][i]):
                max_temp = np.max(freq_clusters[j][i])
        ylim_list.append([0, round_up(max_temp*1.1,0)])      
        
        # add the relationship to artificial-stimuli cluster if fac is false
    if not fac:
        freq_fac = []
        cluster_num_fac = np.unique(df_results['cluster_label_dendro_num']).size
        bins_fac = np.linspace(1,cluster_num_fac+1,cluster_num_fac+1)
        for i in range(fig_row):
            data_fac_i = df_results['cluster_label_dendro_num'][df_results['genetic_label_num']==i]
            hist_fac_i, edges_fac_i = np.histogram(cluster_num_fac-data_fac_i, bins_fac)
            freq_fac.append(hist_fac_i/float(hist_fac_i.sum())*100)
        freq_clusters.append(freq_fac)
        data_fac_all = df_results['cluster_label_dendro_num']
        hist_fac_all, edges_fac_all = np.histogram(cluster_num_fac-data_fac_all, bins_fac)
        freq_fac_all = hist_fac_all/float(hist_fac_all.sum())*100
        
        ylim_fac_all = np.max(freq_fac_all)    
        
        ylim_fac_list = []
        for i in range(fig_row):
            # max_temp = np.max([ylim_fac_all,np.max(freq_fac[i])])
            max_temp = np.max([ylim_fac_all,np.max(freq_fac)]) # set the ylim the same for all rows
            # max_temp = np.max(freq_fac[i])
            ylim_fac_list.append([0, round_up(max_temp*1.1,0)])     
           
    for i in range(fig_row):
        for j in range(fig_col):
            if not fac:
                if j < fig_col-1:
                    w = bins_list[j][1]-bins_list[j][0]
                    axs[i,j].bar(bins_list[j][:-1], freq_all[j], width=w, align="edge", edgecolor="none", color='grey', alpha=alpha)
                    # axs[i,j].axes.xaxis.set_ticklabels([])
                    # axs[i,j].axes.yaxis.set_ticklabels([])
                    
                    axs[i,j].bar(bins_list[j][:-1], freq_clusters[i][j], width=w, align="edge", edgecolor="none", color='green', alpha=alpha)
                    # axs[i,j].axes.xaxis.set_ticklabels([])
                    # axs[i,j].axes.yaxis.set_ticklabels([])
                    axs[i,j].set_ylim(ylim_list[j])
                else:
                    w = bins_fac[1] - bins_fac[0]
                    axs[i,j].bar(bins_fac[:-1], freq_fac_all, width=w, align="edge", edgecolor="none", color='gray', alpha=1)
                    axs[i,j].bar(bins_fac[:-1], freq_fac[i], width=w, align="edge", edgecolor="none", color='green', alpha=1)
                    axs[i,j].set_ylim(ylim_fac_list[i])
                    axs[i,j].set_xticks([0,10,20])
            else:
                w = bins_list[j][1]-bins_list[j][0]
                axs[i,j].bar(bins_list[j][:-1], freq_all[j], width=w, align="edge", edgecolor="none", color='grey', alpha=alpha)               
                axs[i,j].bar(bins_list[j][:-1], freq_clusters[i][j], width=w, align="edge", edgecolor="none", color='green', alpha=alpha)
                axs[i,j].set_ylim(ylim_list[j])
                    
                
            # Hide the right and top spines
            axs[i,j].spines['right'].set_visible(False)
            axs[i,j].spines['top'].set_visible(False)
                    # Only show ticks on the left and bottom spines
            axs[i,j].yaxis.set_ticks_position('left')
            axs[i,j].xaxis.set_ticks_position('bottom')
            # set 2^x as x label for the best size 
            if i == fig_row-1 and j == 5:
                formatter = FuncFormatter(lambda x_val, tick_pos: "$2^{{{:.0f}}}$".format((x_val)))
                axs[i,j].xaxis.set_major_formatter(formatter)
            if i == fig_row-1 and j == 0:
                formatter = FuncFormatter(lambda x_val, tick_pos: "$10^{{{:.0f}}}$".format((x_val)))
                axs[i,j].xaxis.set_major_formatter(formatter)
            if i+1 < fig_row:
                axs[i,j].axis("off")
            if j == 0:
                axs[i,j].set_title(genetic_name[i])
            # if i == 0:
            #     axs[i,j].set_title(columns_name[j])
            if i == fig_row-1:
                if fac:
                    axs[i,j].set_xlabel(columns_name[j])   
                else:
                    if j < fig_col-1:
                       axs[i,j].set_xlabel(columns_name[j])
                    else:
                       axs[i,j].set_xlabel('FAC')
                       axs[i,j].set_yticks([])
            if j==0 and i == fig_row-1:
                axs[i,j].set_ylabel('Relative frequency')
            
            # sns.distplot(df_results.iloc[:,j],ax=axs[i,j], 
            #              bins = bins_list[j], norm_hist=True, kde=False,hist_kws={"color":"blue"})
            # sns.distplot(df_results.loc[df_results['genetic_label_num']==i].iloc[:,j],ax=axs[i,j], 
            #              bins = bins_list[j], norm_hist=True, kde=False,hist_kws={"color":"red"})
    if len(file_save)>0:
        plt.savefig(file_save+'.png',bbox_inches='tight')
        plt.savefig(file_save+'.svg',bbox_inches='tight')
        plt.savefig(file_save+'.pdf',bbox_inches='tight')
    
#%% plot dendrogram for temporal profiles    
def dendrogram_x(x, x_temporal, labels, fig_params_dendro, file_save, method='ward', metric='euclidean',fig_zoom=1,plot_bool=False):
    fig = plt.figure(figsize=[fig_width*fig_zoom,fig_width/3*fig_zoom])
    n_cluster,_ = x.shape
    n_neuron = labels.shape[0]
    axdendro = fig.add_axes(fig_params_dendro['ax_dendro'])
    y = linkage(x, method='ward')
    z = dendrogram(y, orientation='left',distance_sort='descending',show_leaf_counts=True)
    idx = z['leaves']
    axdendro.set_xticks([])
    axdendro.set_yticks([])
    axdendro.axis('off')
    
    labels_dendro = np.zeros_like(labels)
    for i in range(n_cluster):
        labels_dendro[labels==idx[i]] = i
    # count the number of neurons in each cluster
    cluster_size = np.zeros((n_cluster))
    for i in range(n_cluster):
        cluster_size[i] = np.sum(labels_dendro==i)
    

    axdendro_y_min, axdendro_y_max = axdendro.get_ylim()
    axdendro_space_text = (axdendro_y_max - axdendro_y_min) / n_cluster
    for i in range(n_cluster):
        axdendro.text(fig_params_dendro['text_offset'][0],i*axdendro_space_text+fig_params_dendro['text_offset'][1],\
              "{:2d}".format(n_cluster-i) + ' (' + "{:.1f}".format(cluster_size[i]/n_neuron*100) +')')
        # axdendro.text(0.3,i*axdendro_space_text+6,"{:.1f}".format(cluster_size[i]/n_neuron*100))
    
    # Plot distance matrix.
    axmatrix = fig.add_axes(fig_params_dendro['ax_matrix'])
 
    x_temporal_sorted = np.copy(x_temporal[idx,:])
    im = axmatrix.imshow(x_temporal_sorted, aspect='auto', origin='lower',cmap='viridis')
    axmatrix.set_xticks([])
    axmatrix.set_yticks([])
    
    # Plot colorbar.
    axcolor = fig.add_axes(fig_params_dendro['ax_colorbar'])
    plt.colorbar(im, cax=axcolor)
    if not plot_bool:
        plt.close()
    # Display and save figure.
    # plt.show()
    # print(file_save)
    if len(file_save)>0:
        plt.savefig(file_save+'.png',bbox_inches='tight')
        plt.savefig(file_save+'.svg',bbox_inches='tight')
        plt.savefig(file_save+'.pdf',bbox_inches='tight')
    return idx, labels_dendro, cluster_size

#%% plot dendrogram for temporal profiles    
def dendrogram_label(x, labels, method='ward', metric='euclidean'):
    n_cluster,_ = x.shape
    y = linkage(x, method=method,metric=metric)
    z = dendrogram(y, orientation='left',distance_sort='descending',show_leaf_counts=True)
    idx = z['leaves']
    labels_dendro = np.zeros_like(labels)
    for i in range(n_cluster):
        labels_dendro[labels==idx[i]] = i
    # count the number of neurons in each cluster
    cluster_size = np.zeros((n_cluster))
    for i in range(n_cluster):
        cluster_size[i] = np.sum(labels_dendro==i)
    return labels_dendro, cluster_size
#%% plot temporal profiles and histograms for all clusters for artifical stimuli
def figure_cluster_temporal_hist(df_results, cluster_label, x_cluster, cluster_size, 
                                 temporal_list, hist_list, temporal_zooms,
                                 columns_name, bins_list,
                                 file_save, fig_params, fac=True, sampling_rate=5):
    
    cluster_num,feature_col = x_cluster.shape
    num_neuron = np.sum(cluster_size).astype(int)
    fig_row = cluster_num
    fig_col_temporal = len(temporal_list)
    fig_col_hist = len(hist_list)
    if not fac:
        columns_name.append('FAC')
        fig_col_hist = fig_col_hist + 1

    # print(fig_col_hist)
    columns_label = hist_list
    fig_col = fig_col_temporal + fig_col_hist
    fig_col_temporal_size = []
    for i in range(fig_col_temporal):
        _,x_len = temporal_list[i][1].shape
        fig_col_temporal_size.append(x_len*temporal_zooms[i])
    
    # single histogram should have w:h = 16:10
    fig_height_au = cluster_num*(fig_params['height_sub'] + fig_params['space_sub'])
    fig_width_au = np.sum(fig_col_temporal_size) \
                    + fig_col_temporal*fig_params['space_sub'] \
                    + fig_col_hist*(fig_params['width_sub_hist'] + fig_params['space_sub']) 
    # define the left and bottom position and width and height for all subplots
    # relative to figure width and height
    dendro_left = fig_params['margin']
    dendro_bottom = fig_params['margin']
    dendro_width = fig_params['dendro_width']
    dendro_height = 1 - fig_params['margin']*2
    
    subplots_bottom = fig_params['margin'] + (1-2*fig_params['margin'])/fig_row*np.asarray(range(fig_row))
    subplots_width = 1-2*fig_params['margin']-dendro_width
    subplots_left_1 = subplots_width/fig_width_au*(np.cumsum(np.asarray(fig_col_temporal_size)+fig_params['space_sub']))
    subplots_left_2 = subplots_left_1[-1] + subplots_width/fig_width_au*(fig_params['width_sub_hist']+fig_params['space_sub'])*np.asarray(range(fig_col_hist))
    subplots_left = fig_params['offset'] + fig_params['margin'] + dendro_width + np.concatenate((np.asarray([0]),subplots_left_1,subplots_left_2[1:]), axis=0)
    
    subplots_height = (1-2*fig_params['margin'])*fig_params['height_sub']/fig_height_au
    subplots_width_array = subplots_left[1:] - subplots_left[:-1] - subplots_width/fig_width_au*fig_params['space_sub'] 
    #add the width for the last column
    subplots_width_array = np.append(subplots_width_array,subplots_width/fig_width_au*fig_params['width_sub_hist'])
    
    fig_width_dendro = fig_width*fig_params['zoom']
    fig_height_dendro = fig_width_dendro*fig_height_au/(2*fig_width_au)

    
    
    fig = plt.figure(figsize=[fig_width_dendro,fig_height_dendro])
    axdendro = fig.add_axes([dendro_left, dendro_bottom, dendro_width, dendro_height])
    link_cluster = linkage(x_cluster, method='ward')
    dendrogram_cluster = dendrogram(link_cluster, orientation='left',distance_sort='descending',show_leaf_counts=True)
    dendrogram_index = dendrogram_cluster['leaves']
    # x_temporal_sorted = np.copy(x_temporal[dendrogram_index,:])
    axdendro.set_xticks([])
    axdendro.set_yticks([])
    axdendro.axis('off')
    axdendro_y_min, axdendro_y_max = axdendro.get_ylim()
    axdendro_space_text = (axdendro_y_max - axdendro_y_min) / fig_row
    for i in range(fig_row):
        axdendro.text(fig_params['dendro_text_pos'],i*axdendro_space_text+3.8,\
              "{:2d}".format(fig_row-i) + ' (' + "{:.1f}".format(cluster_size[i]/num_neuron*100) +')')
    
    # calculate the relative frequency for histograms
    # for all clusters

    if not fac:
        fig_col_hist_freq = fig_col_hist - 1
    else:
        fig_col_hist_freq = fig_col_hist
    freq_all = []  
    for i in range(fig_col_hist_freq):
        if i == 0 and columns_label[i]=='rf_size':
            hist, edges = np.histogram(np.log10(df_results[columns_label[i]]), bins_list[i])
        else:
            hist, edges = np.histogram(df_results[columns_label[i]], bins_list[i])
        freq_all.append(hist/float(hist.sum())*100)
    # for each cluster
    freq_clusters = []
    for i in range(fig_row):
        freq_temp = []
        for j in range(fig_col_hist_freq):
            if j == 0 and columns_label[j]=='rf_size':
                hist, edges = np.histogram(np.log10(df_results.loc[df_results[cluster_label]==i][columns_label[j]]), bins_list[j])
            else:
                hist, edges = np.histogram(df_results.loc[df_results[cluster_label]==i][columns_label[j]], bins_list[j])              
            freq_temp.append(hist/float(hist.sum())*100)
        freq_clusters.append(freq_temp)
        
    ylim_list = []
    for i in range(fig_col_hist_freq):
        max_temp = np.max(freq_all[i])
        for j in range(fig_row):
            if max_temp < np.max(freq_clusters[j][i]):
                max_temp = np.max(freq_clusters[j][i])
        # print(max_temp)
        ylim_list.append([0, round_up(max_temp*1.1,0)])  
                
    # add the relationship to artificial-stimuli cluster if fac is false
    if not fac:
        freq_fac = []
        cluster_num_fac = np.unique(df_results['cluster_label_dendro_num']).size
        bins_fac = np.linspace(1,cluster_num_fac+1,cluster_num_fac+1)
        for i in range(cluster_num):
            data_fac_i = df_results['cluster_label_dendro_num'][df_results[cluster_label]==i]
            hist_fac_i, edges_fac_i = np.histogram(cluster_num_fac-data_fac_i, bins_fac)
            freq_fac.append(hist_fac_i/float(hist_fac_i.sum())*100)
        freq_clusters.append(freq_fac)
        
        ylim_fac_list = []
        for i in range(cluster_num):
            max_temp = np.max(freq_fac[i])
            ylim_fac_list.append([0, round_up(max_temp*1.1,0)])    
    

  
        
    k = 99
    # for i in range(2):
    #     for j in range(2): 

        
    for i in range(fig_row):
        for j in range(fig_col):
            left = subplots_left[j]
            bottom = subplots_bottom[i]
            width = subplots_width_array[j]
            height = subplots_height
            ax = fig.add_axes([left,bottom,width,height])
            if j<fig_col_temporal:
                y = temporal_list[j][1][dendrogram_index[i],:]
                t = np.asarray(range(1, y.shape[0]+1))/sampling_rate
                ax.plot(t,y, color='black')   
                ax.set_ylim([0, 1.02])       
                # if j==0 and i == 0:
                #     ax.set_ylabel('Normalized amplitude')                      
            else:
                k = j - fig_col_temporal
                # plot histogram for all clusters
                if not fac:
                    if k < fig_col_hist_freq:
                        w = bins_list[k][1]-bins_list[k][0]
                        ax.bar(bins_list[k][:-1], freq_all[k], width=w, align="edge", edgecolor="none", color='grey', alpha=1)
                    
                        # plot histogram for neurons in a specific cluster
                        ax.bar(bins_list[k][:-1], freq_clusters[i][k], width=w, align="edge", edgecolor="none", color='green', alpha=1)
                        ax.set_ylim(ylim_list[k])
                    else:
                        w = bins_fac[1] - bins_fac[0]
                        ax.bar(bins_fac[:-1], freq_fac[i], width=w, align="edge", edgecolor="none", color='green', alpha=1)
                        ax.set_ylim(ylim_fac_list[i])
                        ax.set_xticks([0,10,20])
                else:
                    # print(k)
                    # print(j)
                    # print(fig_col)
                    # print(fig_col_temporal)
                    w = bins_list[k][1]-bins_list[k][0]
                    ax.bar(bins_list[k][:-1], freq_all[k], width=w, align="edge", edgecolor="none", color='grey', alpha=1)
                
                    # plot histogram for neurons in a specific cluster
                    ax.bar(bins_list[k][:-1], freq_clusters[i][k], width=w, align="edge", edgecolor="none", color='green', alpha=1)
                    ax.set_ylim(ylim_list[k])
                if i == 0:
                    ax.set_xlabel(columns_name[k])
                # if i == 0 and k == 0:
                #     ax.set_ylabel('Relative frequency')
                #     formatter = FuncFormatter(lambda x_val, tick_pos: "$10^{{{:.0f}}}$".format((x_val)))
                #     ax.xaxis.set_major_formatter(formatter)
                # # set 2^x as x label for the best size 
                # if i == 0 and k == 5:
                #     formatter = FuncFormatter(lambda x_val, tick_pos: "$2^{{{:.0f}}}$".format((x_val)))
                #     ax.xaxis.set_major_formatter(formatter)
            
            if i > 0:
                ax.set_xticks([]) 
                ax.set_yticks([]) 
                # ax.axes.xaxis.set_ticklabels([])
                # ax.axes.yaxis.set_ticklabels([])
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                # ax.axis("off")
            else:
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                        # Only show ticks on the left and bottom spines
                ax.yaxis.set_ticks_position('left')
                ax.xaxis.set_ticks_position('bottom')
                ax.set_xticks([]) 
                ax.set_yticks([]) 
    if len(file_save)>0:
        plt.savefig(file_save+'.png',bbox_inches='tight')
        plt.savefig(file_save+'.svg',bbox_inches='tight')
        plt.savefig(file_save+'.pdf',bbox_inches='tight')
    
#%% plot temporal profiles and histograms for all clusters for natural movie
def figure_cluster_nm_temporal_hist(df_results, dendro_cluster, x_cluster, cluster_size, 
                                 temporal_list, hist_list, temporal_zooms,
                                 file_save, fig_params, sampling_rate=5):
    # hist_list is the name in dataframe
    # columns_name is the x_label
    cluster_num,feature_col = x_cluster.shape
    num_neuron = np.sum(cluster_size).astype(int)
    fig_row = cluster_num
    fig_col_temporal = len(temporal_list)
    fig_col_hist = len(hist_list)
    columns_label = hist_list
    fig_col = fig_col_temporal + fig_col_hist
    fig_col_temporal_size = []
    for i in range(fig_col_temporal):
        _,x_len = temporal_list[i][1].shape
        fig_col_temporal_size.append(x_len*temporal_zooms[i])
    
    # single histogram should have w:h = 16:10
    fig_height_au = cluster_num*(fig_params['height_sub'] + fig_params['space_sub'])
    fig_width_au = np.sum(fig_col_temporal_size) \
                    + fig_col_temporal*fig_params['space_sub'] \
                    + fig_col_hist*(fig_params['width_sub_hist'] + fig_params['space_sub']) 
    # define the left and bottom position and width and height for all subplots
    # relative to figure width and height
    dendro_left = fig_params['margin']
    dendro_bottom = fig_params['margin']
    dendro_width = fig_params['dendro_width'] 
    dendro_height = 1 - fig_params['margin']*2
    
    subplots_bottom = fig_params['margin'] + (1-2*fig_params['margin'])/fig_row*np.asarray(range(fig_row))
    subplots_width = 1-2*fig_params['margin']-dendro_width
    subplots_left_1 = subplots_width/fig_width_au*(np.cumsum(np.asarray(fig_col_temporal_size)+fig_params['space_sub']))
    subplots_left_2 = subplots_left_1[-1] + subplots_width/fig_width_au*(fig_params['width_sub_hist']+fig_params['space_sub'])*np.asarray(range(fig_col_hist))
    subplots_left = fig_params['offset'] + dendro_width + np.concatenate((np.asarray([0]),subplots_left_1,subplots_left_2[1:]), axis=0)
    
    subplots_height = (1-2*fig_params['margin'])*fig_params['height_sub']/fig_height_au
    subplots_width_array = subplots_left[1:] - subplots_left[:-1] - subplots_width/fig_width_au*fig_params['space_sub'] 
    #add the width for the last column
    subplots_width_array = np.append(subplots_width_array,subplots_width/fig_width_au*fig_params['width_sub_hist'])
    
    fig_width_dendro = fig_width*fig_params['zoom']
    fig_height_dendro = fig_width_dendro*fig_height_au/(2*fig_width_au)
    
    
    fig = plt.figure(figsize=[fig_width_dendro,fig_height_dendro])
    axdendro = fig.add_axes([dendro_left, dendro_bottom, dendro_width, dendro_height])
    link_cluster = linkage(dendro_cluster, method='ward')
    dendrogram_cluster = dendrogram(link_cluster, orientation='left',distance_sort='descending',show_leaf_counts=True)
    dendrogram_index = dendrogram_cluster['leaves']
    # print(dendrogram_index)
    # x_temporal_sorted = np.copy(x_temporal[dendrogram_index,:])
    axdendro.set_xticks([])
    axdendro.set_yticks([])
    axdendro.axis('off')
    axdendro_y_min, axdendro_y_max = axdendro.get_ylim()
    axdendro_space_text = (axdendro_y_max - axdendro_y_min) / fig_row
    for i in range(fig_row):
        axdendro.text(fig_params['dendro_text_pos'],i*axdendro_space_text+3.8,\
                      "{:2d}".format(fig_row-i) + ' (' + "{:.1f}".format(cluster_size[i]/num_neuron*100) +')')
    
    if fig_col_hist>0:
        # calculate the relative frequency for histograms
        # for all clusters
        freq_all = []
        cluster_num_fac = np.unique(df_results['cluster_label_dendro_num']).size
        bins = np.linspace(1,cluster_num_fac+1,cluster_num_fac+1)
        for i in range(cluster_num):
            data_temp = df_results['cluster_label_dendro_num'][df_results['cluster_label_nm_dendro_num']==i]
            hist, edges = np.histogram(cluster_num_fac-data_temp, bins)
            freq_all.append(hist/float(hist.sum())*100)
        ylim_list = []
        for i in range(cluster_num):
            max_temp = np.max(freq_all[i])
            ylim_list.append([0, round_up(max_temp*1.1,0)])        
    # for i in range(2):
    #     for j in range(2):   
    for i in range(fig_row):
        for j in range(fig_col):
            left = subplots_left[j]
            bottom = subplots_bottom[i]
            width = subplots_width_array[j]
            height = subplots_height
            ax = fig.add_axes([left,bottom,width,height])
            if j<fig_col_temporal:
                y = temporal_list[j][1][dendrogram_index[i],:]
                t = np.asarray(range(1, y.shape[0]+1))/sampling_rate
                ax.plot(t,y, color='black')   
                ax.set_ylim([0, 1.02])  
                dt = t[1] - t[0]
                ax.set_xlim([t[0]-dt, t[-1]+dt]) 
                # if j==0 and i == 0:
                #     ax.set_ylabel('Normalized amplitude')                      
            else:
                # plot histogram for all neurons
                if fig_col_hist>0:
                    w = bins[1]-bins[0]
                    ax.bar(bins[:-1], freq_all[i], width=w, align="edge", edgecolor="none", color='grey', alpha=1)
                    ax.set_ylim(ylim_list[i])
                    ax.set_xticks([0,10,20])
                    if i == 0:
                        ax.set_xlabel(hist_list[0])
                # if i == 0 and k == 0:
                #     ax.set_ylabel('Relative frequency')
                #     formatter = FuncFormatter(lambda x_val, tick_pos: "$10^{{{:.0f}}}$".format((x_val)))
                #     ax.xaxis.set_major_formatter(formatter)
                # # set 2^x as x label for the best size 
                # if i == 0 and k == 5:
                #     formatter = FuncFormatter(lambda x_val, tick_pos: "$2^{{{:.0f}}}$".format((x_val)))
                #     ax.xaxis.set_major_formatter(formatter)
            
            if i > 0:
                ax.set_xticks([]) 
                ax.set_yticks([]) 
                # ax.axes.xaxis.set_ticklabels([])
                # ax.axes.yaxis.set_ticklabels([])
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                # ax.axis("off")
            else:
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                        # Only show ticks on the left and bottom spines
                ax.yaxis.set_ticks_position('left')
                ax.xaxis.set_ticks_position('bottom')
                # ax.set_xticks([]) 
                ax.set_yticks([]) 
    if len(file_save)>0:
        plt.savefig(file_save+'.png',bbox_inches='tight')
        plt.savefig(file_save+'.svg',bbox_inches='tight')
        plt.savefig(file_save+'.pdf',bbox_inches='tight')

#%% scatter plot of rf positions for all cluster
def figure_scatter_rf_pos_cluster(df_rf, label, fig_params, file_save,color_id=None):
    # df_results contains azimuth, elevation, labels
    hwr = fig_params['hwr'] #height to width ratio
    fig_col = fig_params['fig_col'] # set figure columns as 6
    cluster_num = np.unique(df_rf[label]).size
    # print(cluster_num)
    fig_row = np.ceil(cluster_num/fig_col).astype(int)
    fig = plt.figure(figsize=(fig_width*fig_params['zoom'],fig_width*fig_params['zoom']/fig_col*fig_row*hwr))
    # print(fig_col)
    # print(fig_row)
    subplot_height_space = (1-2*fig_params['margin'])/fig_row
    subplot_width_space = (1-2*fig_params['margin'])/fig_col
    
    subplots_bottom = fig_params['margin'] + subplot_height_space*np.asarray(range(fig_row))
    subplots_left = fig_params['margin'] + subplot_width_space*np.asarray(range(fig_col)) 
    subplots_height = subplot_height_space - fig_params['space']
    subplots_width = subplot_width_space - fig_params['space']
    
    xlim_min = round_down(np.min(df_rf['rf_azimuth']),-1)
    xlim_max =  round_up(np.max(df_rf['rf_azimuth']),-1)
    xlim_range = xlim_max - xlim_min 
    xlim = [xlim_min, xlim_max]
    ylim_min = round_down(np.min(df_rf['rf_elevation']),-1)
    ylim_max = round_up(np.max(df_rf['rf_elevation']),-1)
    ylim = [ylim_min, ylim_max]
    ylim_range = ylim_max - ylim_min 
    # print(xlim)
    # print(ylim)
    # for i in range(2):
    #     for j in range(2):   
  
    # color_id = 'mouse_id'

    genetic_id_unique = np.unique(df_rf['genetic_label_num'])
    num_genetic = genetic_id_unique.size
    if color_id == 'mouse_id':
        cmap = plt.get_cmap('jet')
        mouse_id_unique = np.unique(df_rf['mouse_id'])
        num_animal = mouse_id_unique.size
        num_max_color = num_animal
    elif color_id =='genetic_id':
        cmap = plt.get_cmap('viridis')
        genetic_id_unique = np.unique(df_rf['genetic_label_num'])
        num_genetic = genetic_id_unique.size
        num_max_color = num_genetic
    else:
        cmap = plt.get_cmap('jet')
        num_max_color = 18
    colors = [cmap(i) for i in np.linspace(0, 1, num_max_color)]
    # colors = [cmap(i) for i in np.linspace(0, 1, num_max_color)] #18 is the number of maximal color number
    if color_id != 'genetic_id':
        colors = colors[::-1]
    # print(len(colors))
    for i in range(fig_row):
        for j in range(fig_col):
            left = subplots_left[j]
            bottom = subplots_bottom[fig_row-i-1]
            width = subplots_width
            height = subplots_height
            k = i*fig_col + j
            k_rev = cluster_num-k-1
            azi = df_rf['rf_azimuth'][df_rf[label]==k_rev]
            ele = df_rf['rf_elevation'][df_rf[label]==k_rev]
            img_id = df_rf['image_id'][df_rf[label]==k_rev]
            # print(img_id.shape)
            # print(np.unique(img_id).shape)
            if color_id == 'mouse_id':
                animal_id = df_rf['mouse_id'][df_rf[label]==k_rev]
                # print(animal_id.shape)
                # print(np.unique(animal_id).shape)
            elif color_id == 'genetic_id':
                genetic_id = df_rf['genetic_label_num'][df_rf[label]==k_rev] 
            # img_id = df_rf[color_id][df_rf[label]==k_rev]
            if len(azi)>0:                
                ax = fig.add_axes([left,bottom,width,height])
                # plot rf positions of all neurons
                # ax.scatter(df_rf['rf_azimuth'], df_rf['rf_elevation'], s=1, color='grey',alpha=0.1)
                # ax.set_aspect('equal')
                # plot the median position
                # ax.scatter(np.nanmedian(df_rf['rf_azimuth']), np.nanmedian(df_rf['rf_elevation']), s=20, color='black',marker='x',alpha=1)
                ax.plot([xlim_min,xlim_max],[np.nanmedian(df_rf['rf_elevation']),np.nanmedian(df_rf['rf_elevation'])],color='red',linewidth=0.2)
                ax.plot([np.nanmedian(df_rf['rf_azimuth']),np.nanmedian(df_rf['rf_azimuth'])],[ylim_min,ylim_max],color='red',linewidth=0.2)
                ax.set_aspect('equal')
                img_id_unique = np.unique(img_id)
                cnt = 0
                for img in img_id_unique:
                    if np.sum(img_id==img)>5:
                        cnt = cnt+1
                # print(cnt)
                # print('ok')
                if cnt>0:
                    if color_id == None:
                        if cnt>num_max_color:
                            print('cnt>num_max_color')
                            cnt = num_max_color
                        colors_sel = colors[::int(num_max_color/cnt)]
                    cnt = 0       
                    for img in img_id_unique:
                        if np.sum(img_id==img)>5:
                            # np.sum(colors_sel[np.mod(cnt,num_max_color)])
                            if color_id == 'mouse_id':
                                _mouse_id = int(np.unique(animal_id[img_id==img]))
                                # print('mouse_id'+str(_mouse_id))
                                _idx_mouse_id = np.where(mouse_id_unique ==_mouse_id)[0][0]
                                ax.scatter(azi[img_id==img], ele[img_id==img], s=2, color=colors[_idx_mouse_id])
                            elif color_id == 'genetic_id':
                                _genetic_id = int(np.unique(genetic_id[img_id==img]))
                                # print('genetic_id'+str(_genetic_id))
                                _idx_genetic_id = np.where(genetic_id_unique==_genetic_id)[0][0]
                                ax.scatter(azi[img_id==img], ele[img_id==img], s=2, color=colors[_idx_genetic_id]) 
                            else:
                                ax.scatter(azi[img_id==img], ele[img_id==img], s=2, color=colors_sel[np.mod(cnt,num_max_color)])
                            cnt = cnt+1
                else:
                    # print('cnt<0')
                    colors_sel = colors[0]
                    ax.scatter(azi[img_id==img], ele[img_id==img], s=1, color=colors_sel)
                ax.set_xlim(xlim) 
                ax.set_ylim(ylim)
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                # Only show ticks on the left and bottom spines
                ax.yaxis.set_ticks_position('left')
                ax.xaxis.set_ticks_position('bottom')
                # # set ticks and ticklabels invisible
                # ax.set_xticks([]) 
                # ax.set_yticks([]) 
                # ax.axes.xaxis.set_ticklabels([])
                # ax.axes.yaxis.set_ticklabels([])
                ax.grid()
                ax.text(xlim_min+xlim_range/20,ylim_min+ylim_range/20,str(k+1),size='xx-large',weight='bold',color='red')
                # print(str(xlim_min)+','+str(xlim_max))
            # ax.axis("off")
    if len(file_save)>0:
        plt.savefig(file_save+'.png',bbox_inches='tight')
        plt.savefig(file_save+'.svg',bbox_inches='tight')
        plt.savefig(file_save+'.pdf',bbox_inches='tight')

#%%        
def temporal_list_generate(X_temporal_cluster,X_temporal_cluster_error, len_list):
    cluster_num,_ = X_temporal_cluster.shape
   
    # chirp_len, mb_len, dl_lr_len, color_len, st_len = len_list # this is before 20210403
    mb_len, dl_lr_len, chirp_len, st_len, color_len = len_list
    dl_len = int(dl_lr_len/2)
    i = 0
    mb_temporal_cluster = X_temporal_cluster[:,i:i+mb_len]
    mb_temporal_cluster_error = X_temporal_cluster_error[:,i:i+mb_len]
    i = i + mb_len
    dl_lr_temporal_cluster = X_temporal_cluster[:,i:i+dl_lr_len]
    dl_lr_temporal_cluster_error = X_temporal_cluster_error[:,i:i+dl_lr_len]
    i = i + dl_lr_len
    chirp_temporal_cluster = X_temporal_cluster[:,i:i+chirp_len]
    chirp_temporal_cluster_error = X_temporal_cluster_error[:,i:i+chirp_len]
    i = i + chirp_len
    st_temporal_cluster = X_temporal_cluster[:,i:i+st_len]
    st_temporal_cluster_error = X_temporal_cluster_error[:,i:i+st_len]
    i = i + st_len
    color_temporal_cluster = X_temporal_cluster[:,i:i+color_len]
    color_temporal_cluster_error = X_temporal_cluster_error[:,i:i+color_len]
    i = i + color_len
    dl_temporal_cluster = dl_lr_temporal_cluster[:,:dl_len]
    lr_temporal_cluster = dl_lr_temporal_cluster[:,dl_len:]
    dl_temporal_cluster_error = dl_lr_temporal_cluster_error[:,:dl_len]
    lr_temporal_cluster_error = dl_lr_temporal_cluster_error[:,dl_len:]
    temporal_list = [
                 ('mb_temporal', mb_temporal_cluster,mb_temporal_cluster_error),
                 ('dl_temporal', dl_temporal_cluster,dl_temporal_cluster_error),
                 ('lr_temporal', lr_temporal_cluster,lr_temporal_cluster_error),
                 ('chirp_temporal', chirp_temporal_cluster,chirp_temporal_cluster_error),
                 ('st_temporal', st_temporal_cluster,st_temporal_cluster_error),
                 ('color_temporal', color_temporal_cluster,color_temporal_cluster_error)]
    return temporal_list

#%% plot temporal profiles and histograms for all clusters for artifical stimuli
def figure_cluster_genetic_temporal_hist(df_results, cluster_label, x_cluster,
                                 temporal_list, hist_list, temporal_zooms,
                                 columns_name, bins_list,
                                 file_save, fig_params, sampling_rate=5):
    
    cluster_num,feature_col = x_cluster.shape
    cluster_size = np.zeros(cluster_num)
    for i in range(cluster_num):
        cluster_size[i] = np.sum(cluster_label == i)
    # num_neuron = np.sum(cluster_size).astype(int)
    fig_row = cluster_num
    fig_col_temporal = len(temporal_list)
    hist_list.append('FAC')
    columns_name.append('FAC')
    fig_col_hist = len(hist_list)
    columns_label = hist_list
    fig_col = fig_col_temporal + fig_col_hist
    fig_col_temporal_size = []
    for i in range(fig_col_temporal):
        _,x_len = temporal_list[i][1].shape
        fig_col_temporal_size.append(x_len*temporal_zooms[i])
    
    # single histogram should have w:h = 16:10
    fig_height_au = cluster_num*(fig_params['height_sub'] + fig_params['space_sub'])
    fig_width_au = np.sum(fig_col_temporal_size) \
                    + fig_col_temporal*fig_params['space_sub'] \
                    + fig_col_hist*(fig_params['width_sub_hist'] + fig_params['space_sub']) 
    # define the left and bottom position and width and height for all subplots
    # relative to figure width and height

    
    subplots_bottom = fig_params['margin'] + (1-2*fig_params['margin'])/fig_row*np.asarray(range(fig_row))
    subplots_width = 1-2*fig_params['margin']
    subplots_left_1 = subplots_width/fig_width_au*(np.cumsum(np.asarray(fig_col_temporal_size)+fig_params['space_sub']))
    subplots_left_2 = subplots_left_1[-1] + subplots_width/fig_width_au*(fig_params['width_sub_hist']+fig_params['space_sub'])*np.asarray(range(fig_col_hist))
    subplots_left = 0.03 + fig_params['margin'] + np.concatenate((np.asarray([0]),subplots_left_1,subplots_left_2[1:]), axis=0)
    
    subplots_height = (1-2*fig_params['margin'])*fig_params['height_sub']/fig_height_au
    subplots_width_array = subplots_left[1:] - subplots_left[:-1] - subplots_width/fig_width_au*fig_params['space_sub'] 
    #add the width for the last column
    subplots_width_array = np.append(subplots_width_array,subplots_width/fig_width_au*fig_params['width_sub_hist'])
    
    fig_width_cluster = fig_width*fig_params['zoom']
    fig_height_cluster = fig_width_cluster*fig_height_au/(2*fig_width_au)

    
    
    fig = plt.figure(figsize=[fig_width_cluster,fig_height_cluster])
    
    # for i in range(fig_row):
    #     axdendro.text(fig_params['dendro_text_pos'],i*axdendro_space_text+3.8,\
    #           "{:2d}".format(fig_row-i) + ' (' + "{:.1f}".format(cluster_size[i]/num_neuron*100) +')')
    
    # calculate the relative frequency for histograms
    # for all clusters
    
    
    freq_all = []
    for i in range(fig_col_hist-1):
        if i == 0 and columns_label[i]=='rf_size':
            hist, edges = np.histogram(np.log10(df_results[columns_label[i]]), bins_list[i])
        else:
            hist, edges = np.histogram(df_results[columns_label[i]], bins_list[i])
        freq_all.append(hist/float(hist.sum())*100)
    # for each cluster
    freq_clusters = []
    for i in range(fig_row):
        freq_temp = []
        for j in range(fig_col_hist-1):
            if j == 0  and columns_label[j]=='rf_size':
                hist, edges = np.histogram(np.log10(df_results.loc[df_results[cluster_label]==i][columns_label[j]]), bins_list[j])
            else:
                hist, edges = np.histogram(df_results.loc[df_results[cluster_label]==i][columns_label[j]], bins_list[j])              
            freq_temp.append(hist/float(hist.sum())*100)
        freq_clusters.append(freq_temp)
        
    freq_fac = []
    cluster_num_fac = np.unique(df_results['cluster_label_dendro_num']).size
    bins_fac = np.linspace(1,cluster_num_fac+1,cluster_num_fac+1)
    for i in range(cluster_num):
        data_fac_i = df_results['cluster_label_dendro_num'][df_results[cluster_label]==i]
        hist_fac_i, edges_fac_i = np.histogram(cluster_num_fac-data_fac_i, bins_fac)
        freq_fac.append(hist_fac_i/float(hist_fac_i.sum())*100)
    freq_clusters.append(freq_fac)
    
    ylim_list = []
    for i in range(fig_col_hist-1):
        max_temp = np.max(freq_all[i])
        for j in range(fig_row):
            if max_temp < np.max(freq_clusters[j][i]):
                max_temp = np.max(freq_clusters[j][i])
        ylim_list.append([0, round_up(max_temp*1.1,0)])  
        
    ylim_fac_list = []
    for i in range(cluster_num):
        max_temp = np.max(freq_fac[i])
        ylim_fac_list.append([0, round_up(max_temp*1.1,0)])   
           
    
    k = 99
    # for i in range(2):
    #     for j in range(2): 
    for i in range(fig_row):
        for j in range(fig_col):
            left = subplots_left[j]
            bottom = subplots_bottom[fig_row-i-1]
            width = subplots_width_array[j]
            height = subplots_height
            ax = fig.add_axes([left,bottom,width,height])
            if j<fig_col_temporal:
                y = temporal_list[j][1][i,:]
                t = np.asarray(range(1, y.shape[0]+1))/sampling_rate
                ax.plot(t,y, color='black')   
                ax.set_ylim([0, 1.02])       
                # if j==0 and i == 0:
                #     ax.set_ylabel('Normalized amplitude')                      
            else:
                k = j - fig_col_temporal
                if k < fig_col_hist-1:
                    # plot histogram for all neurons
                    w = bins_list[k][1]-bins_list[k][0]
                    ax.bar(bins_list[k][:-1], freq_all[k], width=w, align="edge", edgecolor="none", color='grey', alpha=1)
                    
                    # plot histogram for neurons in a specific cluster
                       
    
                    ax.bar(bins_list[k][:-1], freq_clusters[i][k], width=w, align="edge", edgecolor="none", color='green', alpha=1)
                    ax.set_ylim(ylim_list[k])
                else:
                    w = bins_fac[1] - bins_fac[0]
                    ax.bar(bins_fac[:-1], freq_fac[i], width=w, align="edge", edgecolor="none", color='green', alpha=1)
                    ax.set_ylim(ylim_fac_list[i])
                    ax.set_xticks([0,10,20])
                
                if i == 0:
                    ax.set_xlabel(columns_name[k])
                # if i == 0 and k == 0:
                #     ax.set_ylabel('Relative frequency')
                #     formatter = FuncFormatter(lambda x_val, tick_pos: "$10^{{{:.0f}}}$".format((x_val)))
                #     ax.xaxis.set_major_formatter(formatter)
                # # set 2^x as x label for the best size 
                # if i == 0 and k == 5:
                #     formatter = FuncFormatter(lambda x_val, tick_pos: "$2^{{{:.0f}}}$".format((x_val)))
                #     ax.xaxis.set_major_formatter(formatter)
            
            if i > 0:
                ax.set_xticks([]) 
                ax.set_yticks([]) 
                # ax.axes.xaxis.set_ticklabels([])
                # ax.axes.yaxis.set_ticklabels([])
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                # ax.axis("off")
            else:
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                        # Only show ticks on the left and bottom spines
                ax.yaxis.set_ticks_position('left')
                ax.xaxis.set_ticks_position('bottom')
                ax.set_xticks([]) 
                ax.set_yticks([]) 
    if len(file_save)>0:
        plt.savefig(file_save+'.png',bbox_inches='tight')
        plt.savefig(file_save+'.svg',bbox_inches='tight')
        plt.savefig(file_save+'.pdf',bbox_inches='tight')
    return freq_fac
#%%       
def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)
    
#%% plot temporal profiles and functional properties for all clusters for artifical stimuli
def figure_cluster_temporal_box(df_results, cluster_label, x_cluster, cluster_size, 
                                 temporal_list, hist_list, temporal_zooms,
                                 columns_name, bins_list,
                                 file_save, fig_params, fac=True, genetic_comp=False, 
                                 sampling_rate=5,violin=True,norm_flag=False,error=False,box_flag=[],dendro_plot=True):
    # fac: add the relationship to artificial-stimuli cluster if fac is false
    # genetic_comp: plot the genetic composition for each cluster
    cluster_num,feature_col = x_cluster.shape
    num_neuron = np.sum(cluster_size).astype(int)
    fig_row = cluster_num
    fig_col_temporal = len(temporal_list)
    fig_col_hist = len(hist_list)
    if not fac:
        columns_name.append('FAC')
        fig_col_hist = fig_col_hist + 1
        
    if genetic_comp:
        columns_name.append('genetic_comp')
        fig_col_hist = fig_col_hist + 1
    # print(fig_col_hist)
    columns_label = hist_list
    fig_col = fig_col_temporal + fig_col_hist
    fig_col_temporal_size = []
    for i in range(fig_col_temporal):
        _,x_len = temporal_list[i][1].shape
        fig_col_temporal_size.append(x_len*temporal_zooms[i])
    
    # single histogram should have w:h = 16:10
    fig_height_au = cluster_num*(fig_params['height_sub'] + fig_params['space_sub'])
    fig_width_au = np.sum(fig_col_temporal_size) \
                    + fig_col_temporal*fig_params['space_sub'] \
                    + fig_col_hist*(fig_params['width_sub_hist'] + fig_params['space_sub']) 
    # define the left and bottom position and width and height for all subplots
    # relative to figure width and height
    dendro_left = fig_params['margin']
    dendro_bottom = fig_params['margin']
    dendro_width = fig_params['dendro_width']
    dendro_height = 1 - fig_params['margin']*2
    
    subplots_bottom = fig_params['margin'] + (1-2*fig_params['margin'])/fig_row*np.asarray(range(fig_row))
    subplots_width = 1-2*fig_params['margin']-dendro_width
   
    subplots_left_1 = subplots_width/fig_width_au*(np.cumsum(np.asarray(fig_col_temporal_size)+fig_params['space_sub']))
    if len(subplots_left_1)>0:
        subplots_left_2 = subplots_left_1[-1] + subplots_width/fig_width_au*(fig_params['width_sub_hist']+fig_params['space_sub'])*np.asarray(range(fig_col_hist))
    else:
        subplots_left_2 = subplots_width/fig_width_au*(fig_params['width_sub_hist']+fig_params['space_sub'])*np.asarray(range(fig_col_hist))
    subplots_left = fig_params['offset'] + fig_params['margin'] + dendro_width + np.concatenate((np.asarray([0]),subplots_left_1,subplots_left_2[1:]), axis=0)
    
    subplots_height = (1-2*fig_params['margin'])*fig_params['height_sub']/fig_height_au
    subplots_width_array = subplots_left[1:] - subplots_left[:-1] - subplots_width/fig_width_au*fig_params['space_sub'] 
    #add the width for the last column
    subplots_width_array = np.append(subplots_width_array,subplots_width/fig_width_au*fig_params['width_sub_hist'])
    
    fig_width = 8
    fig_width_dendro = fig_width*fig_params['zoom']
    fig_height_dendro = fig_width_dendro*fig_height_au/(2*fig_width_au)

    
    
    fig = plt.figure(figsize=[fig_width_dendro,fig_height_dendro])
    axdendro = fig.add_axes([dendro_left, dendro_bottom, dendro_width, dendro_height])
    link_cluster = linkage(x_cluster, method='ward')
    dendrogram_cluster = dendrogram(link_cluster, orientation='left',distance_sort='descending',show_leaf_counts=True,no_plot=not dendro_plot)
    dendrogram_index = dendrogram_cluster['leaves']
    # x_temporal_sorted = np.copy(x_temporal[dendrogram_index,:])
    axdendro.set_xticks([])
    axdendro.set_yticks([])
    axdendro.axis('off')
    axdendro_y_min, axdendro_y_max = axdendro.get_ylim()
    axdendro_space_text = (axdendro_y_max - axdendro_y_min) / fig_row
    for i in range(fig_row):
        # axdendro.text(fig_params['dendro_text_pos'],i*axdendro_space_text+3.8,\
        #       "{:2d}".format(fig_row-i) + ' (' + "{:.1f}".format(cluster_size[i]/num_neuron*100) +')',fontsize=18)
        axdendro.text(fig_params['dendro_text_pos'],i*axdendro_space_text+3.8,"{:2d}".format(fig_row-i),fontsize=18)
        axdendro.text(fig_params['dendro_text_pos']+1.9,i*axdendro_space_text+0.8,
                      ' (' + "{:.1f}".format(cluster_size[i]/num_neuron*100) +')',fontsize=18)
    
    
    # make the list for data for all clusters

    fig_col_hist_freq = fig_col_hist
    if not fac:
        fig_col_hist_freq = fig_col_hist_freq - 1

        
    if genetic_comp:
        fig_col_hist_freq = fig_col_hist_freq - 1

        
    data_box_all = []  
    for i in range(fig_col_hist_freq):
        data_box_temp = df_results[columns_label[i]]
        data_box_all.append(data_box_temp)
    # for each cluster
    data_box_clusters = []
    for i in range(fig_row):
        data_box_list_temp = []
        for j in range(fig_col_hist_freq):
            data_box_temp = df_results.loc[df_results[cluster_label]==i][columns_label[j]]             
            data_box_list_temp.append(data_box_temp)
        data_box_clusters.append(data_box_list_temp)
        
    freq_all = []
    for i in range(fig_col_hist_freq):
        hist, edges = np.histogram(df_results[columns_label[i]], bins_list[i])
        freq_all.append(hist/float(hist.sum())*100)
    # for each cluster
    freq_clusters = []
    for i in range(fig_row):
        freq_temp = []
        for j in range(fig_col_hist_freq):
            hist, edges = np.histogram(df_results.loc[df_results[cluster_label]==i][columns_label[j]], bins_list[j])              
            freq_temp.append(hist/float(hist.sum())*100)
        freq_clusters.append(freq_temp)
     
    
    ylim_list = []
    for i in range(fig_col_hist_freq):
        max_temp = np.max(freq_all[i])
        for j in range(fig_row):
            if max_temp < np.max(freq_clusters[j][i]):
                max_temp = np.max(freq_clusters[j][i])
        ylim_list.append([0, round_up(max_temp*1.1,0)])  
                
    # add the relationship to artificial-stimuli cluster if fac is false
    if not fac:
        freq_fac = []
        cluster_num_fac = np.unique(df_results['cluster_label_dendro_num']).size
        bins_fac = np.linspace(1,cluster_num_fac+1,cluster_num_fac+1)
        for i in range(cluster_num):
            data_fac_i = df_results['cluster_label_dendro_num'][df_results[cluster_label]==i]
            hist_fac_i, edges_fac_i = np.histogram(cluster_num_fac-data_fac_i, bins_fac)
            freq_fac.append(hist_fac_i/float(hist_fac_i.sum())*100)
        # freq_clusters.append(freq_fac)
        
        ylim_fac_list = []
        for i in range(cluster_num):
            max_temp = np.max(freq_fac[i])
            ylim_fac_list.append([0, round_up(max_temp*1.1,0)])    
    
    # add the genetic compostion for each cluster
    freq_genetic = []
    if genetic_comp:
        cluster_num_genetic = np.unique(df_results['genetic_label_num']).size-1 #-1 is to remove wild type
        bins_genetic = np.linspace(1,cluster_num_genetic+1,cluster_num_genetic+1)
        for i in range(cluster_num):
            freq_genetic_i = np.zeros(cluster_num_genetic)
            for j in range(cluster_num_genetic):
                freq_genetic_i[j] = (np.sum(np.logical_and([df_results[cluster_label]==i],
                                                           df_results['genetic_label_num']==j+1))
                                     # /np.sum(df_results['genetic_label_num']==j))*100 #this is normalized to genetic type
                                     /(np.sum(df_results[cluster_label]==i)-
                                       np.sum(np.logical_and([df_results[cluster_label]==i],
                                                           df_results['genetic_label_num']==0)))
                                     )*100 #this is normalized to functional type
            freq_genetic.append(freq_genetic_i)
            
        ylim_genetic_list = []        
        max_temp = np.zeros(cluster_num)
        for i in range(cluster_num):
            max_temp[i] = np.max(freq_genetic[i])
            ylim_genetic_list.append([0, round_up(max_temp[i]*1.1,0)])           
        ylim_genetic_all = [0,round_up(np.max(max_temp)*1.1,0)] 

        
    k = 99
    xlim_barplot = []   
    # for i in range(2):
    #     for j in range(2): 
    for i in range(fig_row):
        for j in range(fig_col):
        # for j in range(2): 
            left = subplots_left[j]
            bottom = subplots_bottom[i]
            # bottom = subplots_bottom[fig_row-i-1] # this does not work becasue of the dendrogram
            width = subplots_width_array[j]
            height = subplots_height
            ax = fig.add_axes([left,bottom,width,height])
            if j<fig_col_temporal:
                y = temporal_list[j][1][dendrogram_index[i],:]
                t = np.asarray(range(1, y.shape[0]+1))/sampling_rate
                if norm_flag:
                    ax.plot(t,norm_x(y),color='grey') #also plot normalized temporal proflies
                    # if error:
                    #     y_error = temporal_list[j][2][dendrogram_index[i],:]
                    #     ax.fill_between(t,y-y_error,y+y_error,color='lightgrey')
                    
                else:
                    ax.plot(t,y, color='black')   
                    if error:
                        y_error = temporal_list[j][2][dendrogram_index[i],:]
                        ax.fill_between(t,y-y_error,y+y_error,color='lightgrey')
                ax.set_ylim([0, 1.02])       
                # if j==0 and i == 0:
                #     ax.set_ylabel('Normalized amplitude')                      
            else:
                k = j - fig_col_temporal
                # plot histogram for all clusters
                if k < fig_col_hist_freq:
                    if hist_list[k] == 'rf_size':
                        x_min = np.log10(data_box_all[k]).min()
                        x_max = np.log10(data_box_all[k]).max()
                    else:
                        x_min = data_box_all[k].min()
                        x_max = data_box_all[k].max()
                    x_range = x_max - x_min
                    x_margin = x_range*0.05
                    # w = bins_list[k][1]-bins_list[k][0]
                    # ax.bar(bins_list[k][:-1], freq_all[k], width=w, align="edge", edgecolor="none", color='grey', alpha=1)
                    # plot histogram for neurons in a specific cluster
                    if violin:
                        # _data = data_box_all[k]
                        # _filtered_data = _data[~np.isnan(_data)]
                        # violin_plot(_filtered_data,ax)
                        _data = data_box_clusters[i][k]
                        _filtered_data = _data[~np.isnan(_data)]
                        if hist_list[k] in box_flag:
                            w = bins_list[k][1]-bins_list[k][0]
                            # ax.bar(bins_list[k][:-1]-w/2, freq_all[k], width=w, align="edge", edgecolor="none", color='black', alpha=1)
                            ax.bar(bins_list[k][:-1]-w/2, freq_clusters[i][k], width=w, align="edge", edgecolor="none", color='green', alpha=1)
                            ax.set_ylim(ylim_list[k])
                            x_min = bins_list[k].min()-w/2
                            x_max = bins_list[k].max()-w/2
                            x_range = x_max - x_min
                            x_margin = x_range*0.05
                        else:
                            if hist_list[k] == 'rf_size':
                                violin_plot(np.log10(_filtered_data),ax)
                            else:
                                violin_plot(_filtered_data,ax)
                    else:
                        _data = data_box_clusters[i][k]
                        _filtered_data = _data[~np.isnan(_data)]
                        bp = plt.boxplot(_filtered_data, positions=range(1,2), notch=False, vert=False, whis=(0, 100))
                        set_box_color(bp,'green')
                    if i == 0:
                        _data = data_box_all[k]
                        _filtered_data = _data[~np.isnan(_data)]
                        if violin:
                            # sns.violinplot(x=_filtered_data, positions=np.asarray(range(0,1))+0.5, vert=False, color='grey')
                            if hist_list[k] in box_flag:
                                ax.set_ylim(ylim_list[k])
                            else:
                                ax.set_ylim([0.5,1.5])
                        else:
                            bp = plt.boxplot(_filtered_data, positions=np.asarray(range(0,1))+0.5, notch=False, vert=False, whis=(0, 100))
                            set_box_color(bp,'grey')
                            ax.set_ylim([0.2,1.3])
                        
                        ax.set_xlim([x_min-x_margin,x_max+x_margin])
                        # # xlim_barplot.append(ax.get_xlim())
                        # left_xlim, right_xlim = ax.get_xlim()
                        # xlim_barplot.append([np.min([_filtered_data.min(),left_xlim]),
                        #                      np.max([_filtered_data.max(),right_xlim])])
                        # ax.set_xlim(xlim_barplot[k])

                    else:
                        if violin:
                            if hist_list[k] in box_flag:
                                ax.set_ylim(ylim_list[k])
                            else:
                                ax.set_ylim([0.5,1.5])
                        else:
                            ax.set_ylim([0.45,1.55])
                        # print(str(i))
                        # print(str(k))
                        # print(type(xlim_barplot))
                        # print(xlim_barplot[k])
                        # ax.set_xlim(xlim_barplot[k])
                        ax.set_xlim([x_min-x_margin,x_max+x_margin])
                    # if i == 2 and k == 5:
                    #     print(data_box_clusters[i][k])
                    #     plt.figure()
                    #     plt.hist(data_box_clusters[i][k],color='blue')
                    #     plt.figure()
                    #     plt.hist(data_box_all[k],color='black')
                
                else:
                    if (not fac) and genetic_comp:
                        if k == fig_col_hist_freq:
                            # plot the relationship to FAC
                            w = bins_fac[1] - bins_fac[0]
                            ax.bar(bins_fac[:-1], freq_fac[i], width=w, align="edge", edgecolor="none", color='green', alpha=1)
                            ax.set_ylim(ylim_fac_list[i])
                            ax.set_xticks([0,10,20]) #related to the number of cluster
                        else:
                            w = bins_genetic[1] - bins_genetic[0]
                            ax.bar(bins_genetic[:-1], freq_genetic[i], width=w, align="edge", edgecolor="none", color='green', alpha=1)
                            # ax.set_ylim(ylim_genetic_list[i])
                            ax.set_ylim(ylim_genetic_all)
                            ax.set_xticks([0,3,6]) # related to the number of genetic mice lines
                    else:
                        if not fac:  
                        # plot the relationship to FAC
                            w = bins_fac[1] - bins_fac[0]
                            ax.bar(bins_fac[:-1], freq_fac[i], width=w, align="edge", edgecolor="none", color='green', alpha=1)
                            ax.set_ylim(ylim_fac_list[i])
                            ax.set_xticks([0,10,20]) #related to the number of cluster
                        if genetic_comp:
                            # genetic compostion for each cluster
                            w = bins_genetic[1] - bins_genetic[0]
                            ax.bar(bins_genetic[:-1], freq_genetic[i], width=w, align="edge", edgecolor="none", color='green', alpha=1)
                            # ax.set_ylim(ylim_genetic_list[i])
                            ax.set_ylim(ylim_genetic_all)
                            ax.set_xticks([0,3,6]) # related to the number of genetic mice lines
                    
                    # # print(k)
                    # # print(j)
                    # # print(fig_col)
                    # # print(fig_col_temporal)
                    # # w = bins_list[k][1]-bins_list[k][0]
                    # # ax.bar(bins_list[k][:-1], freq_all[k], width=w, align="edge", edgecolor="none", color='grey', alpha=1)
                    # if i == 0:
                    #     bp = plt.boxplot(data_box_all[k], positions=range(0,1), notch=False, vert=False, whis=(0, 100))
                    #     set_box_color(bp,'grey')
                    # # plot histogram for neurons in a specific cluster
                    # # ax.bar(bins_list[k][:-1], freq_clusters[i][k], width=w, align="edge", edgecolor="none", color='green', alpha=1)
                    # # ax.set_ylim(ylim_list[k])
                    # bp = plt.boxplot(data_box_clusters[i][k], positions=range(1,2), notch=False, vert=False, whis=(0, 100))
                    # set_box_color(bp,'green')
                    # ax.set_ylim([-0.5,1.5])
                if i == 0:
                    ax.set_xlabel(columns_name[k],fontsize=18)
            
            if i == 0:
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                        # Only show ticks on the left and bottom spines
                ax.yaxis.set_ticks_position('left')
                ax.xaxis.set_ticks_position('bottom')
                if k < fig_col_hist_freq:
                    ax.set_yticks([]) 
                    # if k == 0:
                    #     formatter = FuncFormatter(lambda x_val, tick_pos: "$10^{{{:.0f}}}$".format((x_val)))
                    #     ax.xaxis.set_major_formatter(formatter)
                    if hist_list[k] == 'best_size':
                        ax.set_xticks([1,5]) # set x_tick for the best size
                        formatter = FuncFormatter(lambda x_val, tick_pos: "$2^{{{:.0f}}}$".format((x_val)))
                        ax.xaxis.set_major_formatter(formatter)
                    elif hist_list[k] == 'rf_size':
                        ax.set_xticks([2,3]) # set x_tick for the best size
                        formatter = FuncFormatter(lambda x_val, tick_pos: "$10^{{{:.0f}}}$".format((x_val)))
                        ax.xaxis.set_major_formatter(formatter)
                    else:
                        formatter = FuncFormatter(lambda x_val, tick_pos: "{:g}".format((x_val)))
                        ax.xaxis.set_major_formatter(formatter)
                    ax.tick_params(labelsize=18)    
                else:
                    ax.set_xticks([])    
                    ax.set_yticks([]) 
            else:
                ax.set_xticks([]) 
                ax.set_yticks([]) 
                # ax.axes.xaxis.set_ticklabels([])
                # ax.axes.yaxis.set_ticklabels([])
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                # ax.axis("off")
    if len(file_save)>0:
        plt.savefig(file_save+'.png',bbox_inches='tight')
        plt.savefig(file_save+'.svg',bbox_inches='tight')
        plt.savefig(file_save+'.pdf',bbox_inches='tight')
        
    return data_box_all, data_box_clusters, freq_genetic
#%% plot temporal profiles and violin plots of functional properties for all clusters for artifical stimuli
def figure_cluster_temporal_violin(df_results, cluster_label, x_cluster, cluster_size, 
                                 temporal_list, hist_list, temporal_zooms,
                                 columns_name, bins_list,
                                 file_save, fig_params, fac=True, genetic_comp=False, sampling_rate=5,violin=True):

    # fac: add the relationship to artificial-stimuli cluster if fac is false
    # genetic_comp: plot the genetic composition for each cluster
    cluster_num,feature_col = x_cluster.shape
    num_neuron = np.sum(cluster_size).astype(int)
    fig_row = cluster_num
    fig_col_temporal = len(temporal_list)
    fig_col_hist = len(hist_list)
    if not fac:
        columns_name.append('FAC')
        fig_col_hist = fig_col_hist + 1
        
    if genetic_comp:
        columns_name.append('genetic_comp')
        fig_col_hist = fig_col_hist + 1
    # print(fig_col_hist)
    columns_label = hist_list
    fig_col = fig_col_temporal + fig_col_hist
    fig_col_temporal_size = []
    for i in range(fig_col_temporal):
        _,x_len = temporal_list[i][1].shape
        fig_col_temporal_size.append(x_len*temporal_zooms[i])
    
    # single histogram should have w:h = 16:10
    fig_height_au = cluster_num*(fig_params['height_sub'] + fig_params['space_sub'])
    fig_width_au = np.sum(fig_col_temporal_size) \
                    + fig_col_temporal*fig_params['space_sub'] \
                    + fig_col_hist*(fig_params['width_sub_hist'] + fig_params['space_sub']) 
    # define the left and bottom position and width and height for all subplots
    # relative to figure width and height
    dendro_left = fig_params['margin']
    dendro_bottom = fig_params['margin']
    dendro_width = fig_params['dendro_width']
    dendro_height = 1 - fig_params['margin']*2
    
    subplots_bottom = fig_params['margin'] + (1-2*fig_params['margin'])/fig_row*np.asarray(range(fig_row))
    subplots_width = 1-2*fig_params['margin']-dendro_width
    subplots_left_1 = subplots_width/fig_width_au*(np.cumsum(np.asarray(fig_col_temporal_size)+fig_params['space_sub']))
    subplots_left_2 = subplots_left_1[-1] + subplots_width/fig_width_au*(fig_params['width_sub_hist']+fig_params['space_sub'])*np.asarray(range(fig_col_hist))
    subplots_left = fig_params['offset'] + fig_params['margin'] + dendro_width + np.concatenate((np.asarray([0]),subplots_left_1,subplots_left_2[1:]), axis=0)
    
    subplots_height = (1-2*fig_params['margin'])*fig_params['height_sub']/fig_height_au
    subplots_width_array = subplots_left[1:] - subplots_left[:-1] - subplots_width/fig_width_au*fig_params['space_sub'] 
    #add the width for the last column
    subplots_width_array = np.append(subplots_width_array,subplots_width/fig_width_au*fig_params['width_sub_hist'])
    
    fig_width_dendro = fig_width*fig_params['zoom']
    fig_height_dendro = fig_width_dendro*fig_height_au/(2*fig_width_au)

    
    
    fig = plt.figure(figsize=[fig_width_dendro,fig_height_dendro])
    axdendro = fig.add_axes([dendro_left, dendro_bottom, dendro_width, dendro_height])
    link_cluster = linkage(x_cluster, method='ward')
    dendrogram_cluster = dendrogram(link_cluster, orientation='left',distance_sort='descending',show_leaf_counts=True)
    dendrogram_index = dendrogram_cluster['leaves']
    # x_temporal_sorted = np.copy(x_temporal[dendrogram_index,:])
    axdendro.set_xticks([])
    axdendro.set_yticks([])
    axdendro.axis('off')
    axdendro_y_min, axdendro_y_max = axdendro.get_ylim()
    axdendro_space_text = (axdendro_y_max - axdendro_y_min) / fig_row
    for i in range(fig_row):
        axdendro.text(fig_params['dendro_text_pos'],i*axdendro_space_text+3.8,\
              "{:2d}".format(fig_row-i) + ' (' + "{:.1f}".format(cluster_size[i]/num_neuron*100) +')')
    
    
    # make the list for data for all clusters

    fig_col_hist_freq = fig_col_hist
    if not fac:
        fig_col_hist_freq = fig_col_hist_freq - 1

        
    if genetic_comp:
        fig_col_hist_freq = fig_col_hist_freq - 1

        
    data_box_all = []  
    for i in range(fig_col_hist_freq):
        if i == 0 and columns_label[i]=='rf_size':
            data_box_temp_raw = np.log10(df_results[columns_label[i]])
            # data_box_temp_raw = df_results[columns_label[i]]
            data_box_temp = data_box_temp_raw[~np.isnan(data_box_temp_raw)]
        else:
            # print(i)
            data_box_temp = df_results[columns_label[i]]
        data_box_all.append(data_box_temp)
    # for each cluster
    data_box_clusters = []
    for i in range(fig_row):
        data_box_list_temp = []
        for j in range(fig_col_hist_freq):
            if j == 0 and columns_label[j]=='rf_size':
                data_box_temp_raw = np.log10(df_results.loc[df_results[cluster_label]==i][columns_label[j]])
                # data_box_temp_raw = df_results.loc[df_results[cluster_label]==i][columns_label[j]]
                data_box_temp = data_box_temp_raw[~np.isnan(data_box_temp_raw)]
            else:
                data_box_temp = df_results.loc[df_results[cluster_label]==i][columns_label[j]]             
            data_box_list_temp.append(data_box_temp)
        data_box_clusters.append(data_box_list_temp)
        
    # ylim_list = []
    # for i in range(fig_col_hist_freq):
    #     max_temp = np.max(freq_all[i])
    #     for j in range(fig_row):
    #         if max_temp < np.max(freq_clusters[j][i]):
    #             max_temp = np.max(freq_clusters[j][i])
    #     ylim_list.append([0, round_up(max_temp*1.1,0)])  
                
    # add the relationship to artificial-stimuli cluster if fac is false
    if not fac:
        freq_fac = []
        cluster_num_fac = np.unique(df_results['cluster_label_dendro_num']).size
        bins_fac = np.linspace(1,cluster_num_fac+1,cluster_num_fac+1)
        for i in range(cluster_num):
            data_fac_i = df_results['cluster_label_dendro_num'][df_results[cluster_label]==i]
            hist_fac_i, edges_fac_i = np.histogram(cluster_num_fac-data_fac_i, bins_fac)
            freq_fac.append(hist_fac_i/float(hist_fac_i.sum())*100)
        # freq_clusters.append(freq_fac)
        
        ylim_fac_list = []
        for i in range(cluster_num):
            max_temp = np.max(freq_fac[i])
            ylim_fac_list.append([0, round_up(max_temp*1.1,0)])    
    
    # add the genetic compostion for each cluster
    if genetic_comp:
        freq_genetic = []
        cluster_num_genetic = np.unique(df_results['genetic_label_num']).size
        bins_genetic = np.linspace(1,cluster_num_genetic+1,cluster_num_genetic+1)
        for i in range(cluster_num):
            freq_genetic_i = np.zeros(cluster_num_genetic)
            for j in range(cluster_num_genetic):
                freq_genetic_i[j] = (np.sum(np.logical_and([df_results[cluster_label]==i],
                                                           df_results['genetic_label_num']==j))
                                     /np.sum(df_results['genetic_label_num']==j))*100
            freq_genetic.append(freq_genetic_i)
            
        ylim_genetic_list = []        
        max_temp = np.zeros(cluster_num)
        for i in range(cluster_num):
            max_temp[i] = np.max(freq_genetic[i])
            ylim_genetic_list.append([0, round_up(max_temp[i]*1.1,0)])           
        ylim_genetic_all = [0,round_up(np.max(max_temp)*1.1,0)] 

  
        
    k = 99

    xlim_barplot = []   
    # for i in range(2):
    #     for j in range(2): 
    for i in range(fig_row):
        for j in range(fig_col):
        # for j in range(2): 
            left = subplots_left[j]
            bottom = subplots_bottom[i]
            # bottom = subplots_bottom[fig_row-i-1] # this does not work becasue of the dendrogram
            width = subplots_width_array[j]
            height = subplots_height
            ax = fig.add_axes([left,bottom,width,height])
            if j<fig_col_temporal:
                y = temporal_list[j][1][dendrogram_index[i],:]
                t = np.asarray(range(1, y.shape[0]+1))/sampling_rate
                ax.plot(t,y, color='black')   
                ax.set_ylim([0, 1.02])       
                # if j==0 and i == 0:
                #     ax.set_ylabel('Normalized amplitude')                      
            else:
                k = j - fig_col_temporal
                # plot histogram for all clusters
                if k < fig_col_hist_freq:
                    # w = bins_list[k][1]-bins_list[k][0]
                    # ax.bar(bins_list[k][:-1], freq_all[k], width=w, align="edge", edgecolor="none", color='grey', alpha=1)
                    # plot histogram for neurons in a specific cluster
                    if violin:
                        _data = data_box_all[k]
                        _filtered_data = _data[~np.isnan(_data)]
                        sns.violinplot(x=_filtered_data, color='grey')
                        _data = data_box_clusters[i][k]
                        _filtered_data = _data[~np.isnan(_data)]
                        sns.violinplot(x=_filtered_data, color='green')
                    else:
                        _data = data_box_clusters[i][k]
                        _filtered_data = _data[~np.isnan(_data)]
                        bp = plt.boxplot(_filtered_data, positions=range(1,2), notch=False, vert=False, whis=(0, 100))
                        set_box_color(bp,'green')
                    if i == 0:
                        _data = data_box_all[k]
                        _filtered_data = _data[~np.isnan(_data)]
                        if violin:
                            # sns.violinplot(x=_filtered_data, positions=np.asarray(range(0,1))+0.5, vert=False, color='grey')
                            ax.set_ylim([-1,1])
                        else:
                            bp = plt.boxplot(_filtered_data, positions=np.asarray(range(0,1))+0.5, notch=False, vert=False, whis=(0, 100))
                            set_box_color(bp,'grey')
                            ax.set_ylim([0.2,1.3])
                       
                        # xlim_barplot.append(ax.get_xlim())
                        left_xlim, right_xlim = ax.get_xlim()
                        xlim_barplot.append([np.min([_filtered_data.min(),left_xlim]),
                                             np.max([_filtered_data.max(),right_xlim])])
                        ax.set_xlim(xlim_barplot[k])
                    else:
                        if violin:
                            ax.set_ylim([-1,1])
                        else:
                            ax.set_ylim([0.45,1.55])
                        # print(str(i))
                        # print(str(k))
                        # print(type(xlim_barplot))
                        # print(xlim_barplot[k])
                        ax.set_xlim(xlim_barplot[k])
                    # if i == 2 and k == 5:
                    #     print(data_box_clusters[i][k])
                    #     plt.figure()
                    #     plt.hist(data_box_clusters[i][k],color='blue')
                    #     plt.figure()
                    #     plt.hist(data_box_all[k],color='black')
                
                else:
                    if (not fac) and genetic_comp:
                        if k == fig_col_hist_freq:
                            # plot the relationship to FAC
                            w = bins_fac[1] - bins_fac[0]
                            ax.bar(bins_fac[:-1], freq_fac[i], width=w, align="edge", edgecolor="none", color='green', alpha=1)
                            ax.set_ylim(ylim_fac_list[i])
                            ax.set_xticks([0,10,20]) #related to the number of cluster
                        else:
                            w = bins_genetic[1] - bins_genetic[0]
                            ax.bar(bins_genetic[:-1], freq_genetic[i], width=w, align="edge", edgecolor="none", color='green', alpha=1)
                            # ax.set_ylim(ylim_genetic_list[i])
                            ax.set_ylim(ylim_genetic_all)
                            ax.set_xticks([0,3,6]) # related to the number of genetic mice lines
                    else:
                        if not fac:  
                        # plot the relationship to FAC
                            w = bins_fac[1] - bins_fac[0]
                            ax.bar(bins_fac[:-1], freq_fac[i], width=w, align="edge", edgecolor="none", color='green', alpha=1)
                            ax.set_ylim(ylim_fac_list[i])
                            ax.set_xticks([0,10,20]) #related to the number of cluster
                        if genetic_comp:
                            # genetic compostion for each cluster
                            w = bins_genetic[1] - bins_genetic[0]
                            ax.bar(bins_genetic[:-1], freq_genetic[i], width=w, align="edge", edgecolor="none", color='green', alpha=1)
                            # ax.set_ylim(ylim_genetic_list[i])
                            ax.set_ylim(ylim_genetic_all)
                            ax.set_xticks([0,3,6]) # related to the number of genetic mice lines
                    
                    # # print(k)
                    # # print(j)
                    # # print(fig_col)
                    # # print(fig_col_temporal)
                    # # w = bins_list[k][1]-bins_list[k][0]
                    # # ax.bar(bins_list[k][:-1], freq_all[k], width=w, align="edge", edgecolor="none", color='grey', alpha=1)
                    # if i == 0:
                    #     bp = plt.boxplot(data_box_all[k], positions=range(0,1), notch=False, vert=False, whis=(0, 100))
                    #     set_box_color(bp,'grey')
                    # # plot histogram for neurons in a specific cluster
                    # # ax.bar(bins_list[k][:-1], freq_clusters[i][k], width=w, align="edge", edgecolor="none", color='green', alpha=1)
                    # # ax.set_ylim(ylim_list[k])
                    # bp = plt.boxplot(data_box_clusters[i][k], positions=range(1,2), notch=False, vert=False, whis=(0, 100))
                    # set_box_color(bp,'green')
                    # ax.set_ylim([-0.5,1.5])
                if i == 0:
                    ax.set_xlabel(columns_name[k])
                # if i == 0 and k == 0:
                #     ax.set_ylabel('Relative frequency')
                #     formatter = FuncFormatter(lambda x_val, tick_pos: "$10^{{{:.0f}}}$".format((x_val)))
                #     ax.xaxis.set_major_formatter(formatter)
                # # set 2^x as x label for the best size 
                # if i == 0 and k == 5:
                #     formatter = FuncFormatter(lambda x_val, tick_pos: "$2^{{{:.0f}}}$".format((x_val)))
                #     ax.xaxis.set_major_formatter(formatter)
            
            if i == 0:
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                        # Only show ticks on the left and bottom spines
                ax.yaxis.set_ticks_position('left')
                ax.xaxis.set_ticks_position('bottom')
                if k < fig_col_hist_freq:
                    ax.set_yticks([]) 
                    formatter = FuncFormatter(lambda x_val, tick_pos: "{:g}".format((x_val)))
                    ax.xaxis.set_major_formatter(formatter)
                    # if k == 0:
                    #     formatter = FuncFormatter(lambda x_val, tick_pos: "$10^{{{:.0f}}}$".format((x_val)))
                    #     ax.xaxis.set_major_formatter(formatter)
                    # if k == 5:
                    #     formatter = FuncFormatter(lambda x_val, tick_pos: "$2^{{{:.0f}}}$".format((x_val)))
                    #     ax.xaxis.set_major_formatter(formatter)
                else:
                    ax.set_xticks([])    
                    ax.set_yticks([]) 
            else:
                ax.set_xticks([]) 
                ax.set_yticks([]) 
                # ax.axes.xaxis.set_ticklabels([])
                # ax.axes.yaxis.set_ticklabels([])
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                # ax.axis("off")
    if len(file_save)>0:
        plt.savefig(file_save+'.png',bbox_inches='tight')
        plt.savefig(file_save+'.svg',bbox_inches='tight')
        plt.savefig(file_save+'.pdf',bbox_inches='tight')
        
    return data_box_all, data_box_clusters
#%% plot genetica labels
def scatter_plot(x,y,hue,x_lim,y_lim):
    fig = plt.figure(figsize=(10,10))
    axes = plt.axes()
    sns.scatterplot(
        x=x, 
        y=y,
        hue=hue.tolist(),
        # palette=sns.color_palette('bright', 6),
        # data=df_results,
        alpha=0.3
    )
    plt.axis('equal')
    axes.set_xlim(x_lim)
    axes.set_ylim(y_lim)
    
#%%plot temporal profiles and box plots for all clusters for genetic labels
def figure_cluster_genetic_temporal_box(df_results, cluster_label, x_cluster,
                                 temporal_list, columns_label, temporal_zooms,
                                 columns_name, bins_list,
                                 file_save, fig_params, fac=True, sampling_rate=5):
    
    cluster_num,feature_col = x_cluster.shape
    cluster_size = np.zeros(cluster_num)
    for i in range(cluster_num):
        cluster_size[i] = np.sum(cluster_label == i)
    num_neuron = np.sum(cluster_size).astype(int)
    fig_row = cluster_num
    fig_col_temporal = len(temporal_list)
    fig_col_hist = len(columns_label)
    if not fac:
        columns_name.append('FAC')
        fig_col_hist = fig_col_hist + 1

    fig_col = fig_col_temporal + fig_col_hist
    fig_col_temporal_size = []
    for i in range(fig_col_temporal):
        _,x_len = temporal_list[i][1].shape
        fig_col_temporal_size.append(x_len*temporal_zooms[i])
    
    # single histogram should have w:h = 16:10
    fig_height_au = cluster_num*(fig_params['height_sub'] + fig_params['space_sub'])
    fig_width_au = np.sum(fig_col_temporal_size) \
                    + fig_col_temporal*fig_params['space_sub'] \
                    + fig_col_hist*(fig_params['width_sub_hist'] + fig_params['space_sub']) 
    # define the left and bottom position and width and height for all subplots
    # relative to figure width and height
  
    
    subplots_bottom = fig_params['margin'] + (1-2*fig_params['margin'])/fig_row*np.asarray(range(fig_row))
    subplots_width = 1-2*fig_params['margin']
    subplots_left_1 = subplots_width/fig_width_au*(np.cumsum(np.asarray(fig_col_temporal_size)+fig_params['space_sub']))
    subplots_left_2 = subplots_left_1[-1] + subplots_width/fig_width_au*(fig_params['width_sub_hist']+fig_params['space_sub'])*np.asarray(range(fig_col_hist))
    subplots_left = fig_params['offset'] + fig_params['margin'] + np.concatenate((np.asarray([0]),subplots_left_1,subplots_left_2[1:]), axis=0)
    
    subplots_height = (1-2*fig_params['margin'])*fig_params['height_sub']/fig_height_au
    subplots_width_array = subplots_left[1:] - subplots_left[:-1] - subplots_width/fig_width_au*fig_params['space_sub'] 
    #add the width for the last column
    subplots_width_array = np.append(subplots_width_array,subplots_width/fig_width_au*fig_params['width_sub_hist'])
    
    fig_width_cluster = fig_width*fig_params['zoom']
    fig_height_cluster = fig_width_cluster*fig_height_au/(2*fig_width_au)

    
    
    fig = plt.figure(figsize=[fig_width_cluster,fig_height_cluster])

    
    # make the list for data for all clusters

    if not fac:
        fig_col_hist_freq = fig_col_hist - 1
    else:
        fig_col_hist_freq = fig_col_hist
        
    data_box_all = []  
    for i in range(fig_col_hist_freq):
        if i == 0 and columns_label[i]=='rf_size':
            data_box_temp_raw = np.log10(df_results[columns_label[i]])
            # data_box_temp_raw = df_results[columns_label[i]]
            data_box_temp = data_box_temp_raw[~np.isnan(data_box_temp_raw)]
        else:
            data_box_temp = df_results[columns_label[i]]
        data_box_all.append(data_box_temp)
    # for each cluster
    data_box_clusters = []
    for i in range(fig_row):
        data_box_list_temp = []
        for j in range(fig_col_hist_freq):
            if j == 0 and columns_label[j]=='rf_size':
                data_box_temp_raw = np.log10(df_results.loc[df_results[cluster_label]==i][columns_label[j]])
                # data_box_temp_raw = df_results.loc[df_results[cluster_label]==i][columns_label[j]]
                data_box_temp = data_box_temp_raw[~np.isnan(data_box_temp_raw)]
            else:
                data_box_temp = df_results.loc[df_results[cluster_label]==i][columns_label[j]]             
            data_box_list_temp.append(data_box_temp)
        data_box_clusters.append(data_box_list_temp)
        
    # ylim_list = []
    # for i in range(fig_col_hist_freq):
    #     max_temp = np.max(freq_all[i])
    #     for j in range(fig_row):
    #         if max_temp < np.max(freq_clusters[j][i]):
    #             max_temp = np.max(freq_clusters[j][i])
    #     ylim_list.append([0, round_up(max_temp*1.1,0)])  
                
    # add the relationship to artificial-stimuli cluster if fac is false
    if not fac:
        freq_fac = []
        cluster_num_fac = np.unique(df_results['cluster_label_dendro_num']).size
        bins_fac = np.linspace(1,cluster_num_fac+1,cluster_num_fac+1)
        for i in range(cluster_num):
            data_fac_i = df_results['cluster_label_dendro_num'][df_results[cluster_label]==i]
            hist_fac_i, edges_fac_i = np.histogram(cluster_num_fac-data_fac_i, bins_fac)
            freq_fac.append(hist_fac_i/float(hist_fac_i.sum())*100)
        data_fac_all = df_results['cluster_label_dendro_num']
        hist_fac_all, edges_fac_all = np.histogram(cluster_num_fac-data_fac_all, bins_fac)
        freq_fac_all = hist_fac_all/float(hist_fac_all.sum())*100
        ylim_fac_all = np.max(freq_fac_all)   
        ylim_fac_list = []
        for i in range(cluster_num):
            max_temp = np.max(freq_fac[i])
            ylim_fac_list.append([0, round_up(max_temp*1.1,0)])    
    else:
        freq_fac_all = []
        freq_fac = []
      
        
    k = 99
    # for i in range(2):
    #     for j in range(2): 

        
    for i in range(fig_row):
        for j in range(fig_col):
            left = subplots_left[j]
            bottom = subplots_bottom[fig_row-i-1]
            width = subplots_width_array[j]
            height = subplots_height
            ax = fig.add_axes([left,bottom,width,height])
            if j<fig_col_temporal:
                y = temporal_list[j][1][i,:]
                ye = temporal_list[j][2][i,:]
                t = np.asarray(range(1, y.shape[0]+1))/sampling_rate
                ax.plot(t,y, color='black')   
                ax.fill_between(t, y-ye, y+ye, color='lightgrey')
                ax.set_ylim([0, 1.02])       
                # if j==0 and i == 0:
                #     ax.set_ylabel('Normalized amplitude')                      
            else:
                k = j - fig_col_temporal
                # plot histogram for all clusters
                if not fac:
                    if k < fig_col_hist_freq:
                        # w = bins_list[k][1]-bins_list[k][0]
                        # ax.bar(bins_list[k][:-1], freq_all[k], width=w, align="edge", edgecolor="none", color='grey', alpha=1)
                        _data = data_box_all[k]
                        _filtered_data = _data[~np.isnan(_data)]
                        bp = plt.boxplot(_filtered_data, positions=range(0,1), notch=False, vert=False, whis=(0, 100))
                        set_box_color(bp,'grey')
                        # plot histogram for neurons in a specific cluster
                        _data = data_box_clusters[i][k]
                        _filtered_data = _data[~np.isnan(_data)]
                        bp = plt.boxplot(_filtered_data, positions=range(1,2), notch=False, vert=False, whis=(0, 100))
                        set_box_color(bp,'green')
                        ax.set_ylim([-0.5,1.5])
                    else:
                        # plot the relationship to FAC
                        w = bins_fac[1] - bins_fac[0]
                        ax.bar(bins_fac[:-1], freq_fac[i], width=w, align="edge", edgecolor="none", color='green', alpha=1)
                        ax.set_ylim(ylim_fac_list[i])
                        # ax.set_xticks([0,10,20])
                else:
                    # print(k)
                    # print(j)
                    # print(fig_col)
                    # print(fig_col_temporal)
                    # w = bins_list[k][1]-bins_list[k][0]
                    # ax.bar(bins_list[k][:-1], freq_all[k], width=w, align="edge", edgecolor="none", color='grey', alpha=1)
                    _data = data_box_all[k]
                    _filtered_data = _data[~np.isnan(_data)]
                    bp = plt.boxplot(_filtered_data, positions=range(0,1), notch=False, vert=False, whis=(0, 100))
                    set_box_color(bp,'grey')
                    # plot histogram for neurons in a specific cluster
                    # ax.bar(bins_list[k][:-1], freq_clusters[i][k], width=w, align="edge", edgecolor="none", color='green', alpha=1)
                    # ax.set_ylim(ylim_list[k])
                    _data = data_box_clusters[i][k]
                    _filtered_data = _data[~np.isnan(_data)]
                    bp = plt.boxplot(_filtered_data, positions=range(1,2), notch=False, vert=False, whis=(0, 100))
                    set_box_color(bp,'green')
                    ax.set_ylim([-0.5,1.5])
                # if i == 0:
                #     ax.set_xlabel(columns_name[k])
                # if i == 0 and k == 0:
                #     ax.set_ylabel('Relative frequency')
                #     formatter = FuncFormatter(lambda x_val, tick_pos: "$10^{{{:.0f}}}$".format((x_val)))
                #     ax.xaxis.set_major_formatter(formatter)
                # # set 2^x as x label for the best size 
                # if i == 0 and k == 5:
                #     formatter = FuncFormatter(lambda x_val, tick_pos: "$2^{{{:.0f}}}$".format((x_val)))
                #     ax.xaxis.set_major_formatter(formatter)
            
            if i > 0:
                ax.set_xticks([]) 
                ax.set_yticks([]) 
                # ax.axes.xaxis.set_ticklabels([])
                # ax.axes.yaxis.set_ticklabels([])
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                # ax.axis("off")
            else:
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                        # Only show ticks on the left and bottom spines
                ax.yaxis.set_ticks_position('left')
                ax.xaxis.set_ticks_position('bottom')
                ax.set_xticks([]) 
                ax.set_yticks([]) 
    if len(file_save)>0:
        plt.savefig(file_save+'.png',bbox_inches='tight')
        plt.savefig(file_save+'.svg',bbox_inches='tight')
        plt.savefig(file_save+'.pdf',bbox_inches='tight')
        
    return freq_fac_all, np.asarray(freq_fac), data_box_all,data_box_clusters


#%% plot box and histograms for all genetically labeled clusters
def figure_box_genetic(df_results, genetic_name, columns_label, columns_name, bins_list,
                        file_save, fac=False, figsize=(16,10), pad=1.0, alpha=1):
    
    fig_row = len(genetic_name)#7 # number of genetic labels
    fig_col = len(columns_label)#6 # number of functional properties
    if not fac:
        fig_col = fig_col+1
    
    # columns_name = ['RF size', 'DSI', 'OSI', 'HI', 'LSI', 'Pref. size', 'SSI']
    fig, axs = plt.subplots(fig_row, fig_col, sharex=False, figsize=figsize)
    fig.tight_layout(pad=pad)
    
    # calculate the relative frequency for histograms
         
    data_box_all = []  
    for i in range(fig_col-1):
        if i == 0 and columns_label[i]=='rf_size':
            data_box_temp_raw = np.log10(df_results[columns_label[i]])
            # data_box_temp_raw = df_results[columns_label[i]]
            data_box_temp = data_box_temp_raw[~np.isnan(data_box_temp_raw)]
        else:
            data_box_temp = df_results[columns_label[i]]
        data_box_all.append(data_box_temp)
    # for each cluster
    data_box_clusters = []
    for i in range(fig_row):
        data_box_list_temp = []
        for j in range(fig_col-1):
            if j == 0 and columns_label[j]=='rf_size':
                data_box_temp_raw = np.log10(df_results.loc[df_results['genetic_label_num']==i][columns_label[j]])
                # data_box_temp_raw = df_results.loc[df_results[cluster_label]==i][columns_label[j]]
                data_box_temp = data_box_temp_raw[~np.isnan(data_box_temp_raw)]
            else:
                data_box_temp = df_results.loc[df_results['genetic_label_num']==i][columns_label[j]]             
            data_box_list_temp.append(data_box_temp)
        data_box_clusters.append(data_box_list_temp)
       
        
        # add the relationship to artificial-stimuli cluster if fac is false
    if not fac:
        freq_fac = []
        cluster_num_fac = np.unique(df_results['cluster_label_dendro_num']).size
        bins_fac = np.linspace(1,cluster_num_fac+1,cluster_num_fac+1)
        for i in range(fig_row):
            data_fac_i = df_results['cluster_label_dendro_num'][df_results['genetic_label_num']==i]
            hist_fac_i, edges_fac_i = np.histogram(cluster_num_fac-data_fac_i, bins_fac)
            freq_fac.append(hist_fac_i/float(hist_fac_i.sum())*100)
        data_fac_all = df_results['cluster_label_dendro_num']
        hist_fac_all, edges_fac_all = np.histogram(cluster_num_fac-data_fac_all, bins_fac)
        freq_fac_all = hist_fac_all/float(hist_fac_all.sum())*100
        
        ylim_fac_all = np.max(freq_fac_all)    
        
        ylim_fac_list = []
        for i in range(fig_row):
            # max_temp = np.max([ylim_fac_all,np.max(freq_fac[i])])
            max_temp = np.max([ylim_fac_all,np.max(freq_fac)]) # set the ylim the same for all rows
            # max_temp = np.max(freq_fac[i])
            ylim_fac_list.append([0, round_up(max_temp*1.1,0)])     
           
    for i in range(fig_row):
        for j in range(fig_col):
            if not fac:
                if j < fig_col-1:
                    bp = axs[i,j].boxplot(data_box_all[j], positions=range(0,1), notch=False, vert=False, whis=(0, 100))
                    set_box_color(bp,'grey')
                    # plot histogram for neurons in a specific cluster
                    bp = axs[i,j].boxplot(data_box_clusters[i][j], positions=range(1,2), notch=False, vert=False, whis=(0, 100))
                    set_box_color(bp,'green')
                    axs[i,j].set_ylim([-0.5,1.5])
                else:
                    w = bins_fac[1] - bins_fac[0]
                    axs[i,j].bar(bins_fac[:-1], freq_fac_all, width=w, align="edge", edgecolor="none", color='gray', alpha=1)
                    axs[i,j].bar(bins_fac[:-1], freq_fac[i], width=w, align="edge", edgecolor="none", color='green', alpha=1)
                    axs[i,j].set_ylim(ylim_fac_list[i])
                    axs[i,j].set_xticks([0,10,20])
            else:
                bp = plt.boxplot(data_box_all[j], positions=range(0,1), notch=False, vert=False, whis=(0, 100))
                set_box_color(bp,'grey')
                # plot histogram for neurons in a specific cluster
                # ax.bar(bins_list[k][:-1], freq_clusters[i][k], width=w, align="edge", edgecolor="none", color='green', alpha=1)
                # ax.set_ylim(ylim_list[k])
                bp = plt.boxplot(data_box_clusters[i][j], positions=range(1,2), notch=False, vert=False, whis=(0, 100))
                set_box_color(bp,'green')
                axs.set_ylim([-0.5,1.5])
                    
                
            # Hide the right and top spines
            axs[i,j].spines['right'].set_visible(False)
            axs[i,j].spines['top'].set_visible(False)
                    # Only show ticks on the left and bottom spines
            axs[i,j].yaxis.set_ticks_position('left')
            axs[i,j].xaxis.set_ticks_position('bottom')
            # set 2^x as x label for the best size 
            if i == fig_row-1 and j == 5:
                formatter = FuncFormatter(lambda x_val, tick_pos: "$2^{{{:.0f}}}$".format((x_val)))
                axs[i,j].xaxis.set_major_formatter(formatter)
            if i == fig_row-1 and j == 0:
                formatter = FuncFormatter(lambda x_val, tick_pos: "$10^{{{:.0f}}}$".format((x_val)))
                axs[i,j].xaxis.set_major_formatter(formatter)
            if i+1 < fig_row:
                axs[i,j].axis("off")
            if j == 0:
                axs[i,j].set_title(genetic_name[i])
            # if i == 0:
            #     axs[i,j].set_title(columns_name[j])
            if i == fig_row-1:
                if fac:
                    axs[i,j].set_xlabel(columns_name[j])   
                else:
                    if j < fig_col-1:
                       axs[i,j].set_xlabel(columns_name[j])
                    else:
                       axs[i,j].set_xlabel('FAC')
                       axs[i,j].set_yticks([])
            if j==0 and i == fig_row-1:
                axs[i,j].set_ylabel('Relative frequency')
            
            # sns.distplot(df_results.iloc[:,j],ax=axs[i,j], 
            #              bins = bins_list[j], norm_hist=True, kde=False,hist_kws={"color":"blue"})
            # sns.distplot(df_results.loc[df_results['genetic_label_num']==i].iloc[:,j],ax=axs[i,j], 
            #              bins = bins_list[j], norm_hist=True, kde=False,hist_kws={"color":"red"})
    if len(file_save)>0:
        plt.savefig(file_save+'.png',bbox_inches='tight')
        plt.savefig(file_save+'.svg',bbox_inches='tight')
        plt.savefig(file_save+'.pdf',bbox_inches='tight')
    return freq_fac_all, np.asarray(freq_fac)

#%% plot violin and histograms for all genetically labeled clusters
def figure_violin_genetic(df_results, genetic_name, df_col, columns_label,  columns_name, bins_list,
                        file_save, fac=False, figsize=(16,10), pad=1.0, alpha=1, box_flag=[],bar_flag=[],fac_plot=False):
    
    fig_row = len(genetic_name)#7 # number of genetic labels
    fig_col = len(columns_label)#6 # number of functional properties
    if not fac:
        fig_col = fig_col+1
    
    # columns_name = ['RF size', 'DSI', 'OSI', 'HI', 'LSI', 'Pref. size', 'SSI']
    fig, axs = plt.subplots(fig_row, fig_col, sharex=False, figsize=figsize)
    fig.tight_layout(pad=pad)
    
    # calculate the relative frequency for histograms
    if not fac:
        fig_col_box = fig_col-1
    else:
        fig_col_box = fig_col
    data_box_all = []  
    for i in range(fig_col_box):
        if columns_label[i]=='rf_size':
            data_box_temp_raw = np.log10(df_results[columns_label[i]])
            # data_box_temp_raw = df_results[columns_label[i]]
            data_box_temp = data_box_temp_raw[~np.isnan(data_box_temp_raw)]
        else:
            data_box_temp = df_results[columns_label[i]]
        data_box_all.append(data_box_temp)
    # for each cluster
    data_box_clusters = []
    for i in range(fig_row):
        data_box_list_temp = []
        for j in range(fig_col_box):
            if columns_label[j]=='rf_size':
                data_box_temp_raw = np.log10(df_results.loc[df_results[df_col]==genetic_name[i]][columns_label[j]])
                # data_box_temp_raw = df_results.loc[df_results[cluster_label]==i][columns_label[j]]
                data_box_temp = data_box_temp_raw[~np.isnan(data_box_temp_raw)]
            else:
                data_box_temp = df_results.loc[df_results[df_col]==genetic_name[i]][columns_label[j]]             
            data_box_list_temp.append(data_box_temp)
        data_box_clusters.append(data_box_list_temp)

    freq_all = []
    for i in range(fig_col-1):
        hist, edges = np.histogram(df_results[columns_label[i]], bins_list[i])
        freq_all.append(hist/float(hist.sum())*100)
    # for each cluster
    freq_clusters = []
    for i in range(fig_row):
        freq_temp = []
        for j in range(fig_col-1):
            hist, edges = np.histogram(df_results.loc[df_results[df_col]==genetic_name[i]][columns_label[j]], bins_list[j])              
            freq_temp.append(hist/float(hist.sum())*100)
        freq_clusters.append(freq_temp)
     
    
    ylim_list = []
    for i in range(fig_col-1):
        max_temp = np.max(freq_all[i])
        for j in range(fig_row):
            if max_temp < np.max(freq_clusters[j][i]):
                max_temp = np.max(freq_clusters[j][i])
        ylim_list.append([0, round_up(max_temp*1.1,0)])            
        # add the relationship to artificial-stimuli cluster if fac is false
    if not fac:

        freq_fac = []
        cluster_num_fac = np.unique(df_results['cluster_label_dendro_num']).size
        bins_fac = np.linspace(1,cluster_num_fac+1,cluster_num_fac+1)
        for i in range(fig_row):
            data_fac_i = df_results['cluster_label_dendro_num'][df_results[df_col]==genetic_name[i]]
            hist_fac_i, edges_fac_i = np.histogram(cluster_num_fac-data_fac_i, bins_fac)
            freq_fac.append(hist_fac_i/float(hist_fac_i.sum())*100)
        data_fac_all = df_results['cluster_label_dendro_num']
        hist_fac_all, edges_fac_all = np.histogram(cluster_num_fac-data_fac_all, bins_fac)
        freq_fac_all = hist_fac_all/float(hist_fac_all.sum())*100
        
        ylim_fac_all = np.max(freq_fac_all)    
        
        ylim_fac_list = []
        for i in range(fig_row):
            # max_temp = np.max([ylim_fac_all,np.max(freq_fac[i])])
            max_temp = np.max([ylim_fac_all,np.max(freq_fac)]) # set the ylim the same for all rows
            # max_temp = np.max(freq_fac[i])
            ylim_fac_list.append([0, round_up(max_temp*1.1,0)])     
    else:
        freq_fac_all = []
        freq_fac = []
    for i in range(fig_row):
        for j in range(fig_col):
            if not fac:
                if j < fig_col-1:
                    _data = data_box_clusters[i][j]
                    _filtered_data = _data[~np.isnan(_data)]
                    if columns_label[j] in box_flag:
                        w = bins_list[j][1]-bins_list[j][0]
                        # axs[i,j].bar(bins_list[j][:-1]-w/2, freq_all[j], width=w, align="edge", edgecolor="none", color='black', alpha=1)
                        axs[i,j].bar(bins_list[j][:-1]-w/2, freq_clusters[i][j], width=w, align="edge", edgecolor="none", color='green', alpha=1)
                        axs[i,j].set_ylim(ylim_list[j])
                        x_min = bins_list[j].min()-w/2
                        x_max = bins_list[j].max()-w/2
                        x_range = x_max - x_min
                        x_margin = x_range*0.05
                    else:
                        if columns_label[j] == 'rf_size':
                            violin_plot(_filtered_data,axs[i,j])
                            x_min = data_box_all[j].min()
                            x_max = data_box_all[j].max()
                            # violin_plot(np.log10(_filtered_data),axs[i,j])
                            # x_min = np.log10(_filtered_data).min()
                            # x_max = np.log10(_filtered_data).max()
                        else:
                            violin_plot(_filtered_data,axs[i,j])
                            x_min = data_box_all[j].min()
                            x_max = data_box_all[j].max()
                        axs[i,j].set_ylim([0.5,1.5])                    
                        x_range = x_max - x_min
                        x_margin = x_range*0.05
                    axs[i,j].set_xlim([x_min-x_margin,x_max+x_margin])
                else:
                    w = bins_fac[1] - bins_fac[0]
                    # axs[i,j].bar(bins_fac[:-1], freq_fac_all, width=w, align="edge", edgecolor="none", color='gray', alpha=1)
                    axs[i,j].bar(bins_fac[:-1]-w/2, freq_fac[i], width=w, align="edge", edgecolor="black", color='green', alpha=1)
                    axs[i,j].set_ylim(ylim_fac_list[i])
                    axs[i,j].set_xticks([0,10,20])
            else:
                _data = data_box_clusters[i][j]
                _filtered_data = _data[~np.isnan(_data)]
                if columns_label[j] in box_flag:
                    w = bins_list[j][1]-bins_list[j][0]
                    # axs[i,j].bar(bins_list[j][:-1]-w/2, freq_all[j], width=w, align="edge", edgecolor="none", color='black', alpha=1)
                    axs[i,j].bar(bins_list[j][:-1]-w/2, freq_clusters[i][j], width=w, align="edge", edgecolor="black", color='green', alpha=1)
                    axs[i,j].set_ylim(ylim_list[j])
                    x_min = bins_list[j].min()-w/2
                    x_max = bins_list[j].max()-w/2
                    x_range = x_max - x_min
                    x_margin = x_range*0.05
                else:
                    if columns_label[j] == 'rf_size':
                        violin_plot(_filtered_data,axs[i,j])
                        x_min = data_box_all[j].min()
                        x_max = data_box_all[j].max()
                        # violin_plot(np.log10(_filtered_data),axs[i,j])
                        # x_min = np.log10(_filtered_data).min()
                        # x_max = np.log10(_filtered_data).max()
                        # print(np.mean(_filtered_data))
                    else:
                        violin_plot(_filtered_data,axs[i,j])
                        x_min = data_box_all[j].min()
                        x_max = data_box_all[j].max()
                    axs[i,j].set_ylim([0.5,1.5])                    
                    x_range = x_max - x_min
                    x_margin = x_range*0.05
                axs[i,j].set_xlim([x_min-x_margin,x_max+x_margin])
                    
                
            # Hide the right and top spines
            axs[i,j].spines['right'].set_visible(False)
            axs[i,j].spines['top'].set_visible(False)
                    # Only show ticks on the left and bottom spines
            axs[i,j].yaxis.set_ticks_position('left')
            axs[i,j].xaxis.set_ticks_position('bottom')
            # set 2^x as x label for the best size 
            # if i == fig_row-1 and j == 5:
            #     formatter = FuncFormatter(lambda x_val, tick_pos: "$2^{{{:.0f}}}$".format((x_val)))
            #     axs[i,j].xaxis.set_major_formatter(formatter)
            # if i == fig_row-1 and j == 0:
            #     formatter = FuncFormatter(lambda x_val, tick_pos: "$10^{{{:.0f}}}$".format((x_val)))
            #     axs[i,j].xaxis.set_major_formatter(formatter)
            if i+1 < fig_row:
                # axs[i,j].axis("off")
                axs[i,j].set_xticks([])
                axs[i,j].set_yticks([])
            # if j == 0:
            #     axs[i,j].set_title(genetic_name[i])
            # if i == 0:
            #     axs[i,j].set_title(columns_name[j])
            if i == fig_row-1:
                if fac:
                    if columns_label[j] == 'best_size':
                        axs[i,j].set_xticks([1,5]) # set x_tick for the best size
                        formatter = FuncFormatter(lambda x_val, tick_pos: "$2^{{{:.0f}}}$".format((x_val)))
                        axs[i,j].xaxis.set_major_formatter(formatter)
                    elif columns_label[j] == 'rf_size':
                        axs[i,j].set_xticks([2,3]) # set x_tick for rf size
                        formatter = FuncFormatter(lambda x_val, tick_pos: "$10^{{{:.0f}}}$".format((x_val)))
                        axs[i,j].xaxis.set_major_formatter(formatter)
                    else:
                        formatter = FuncFormatter(lambda x_val, tick_pos: "{:g}".format((x_val)))
                        axs[i,j].xaxis.set_major_formatter(formatter)
                    
                    axs[i,j].tick_params(axis='both', which='major', labelsize=18)
                    axs[i,j].set_yticks([])
                    axs[i,j].set_xlabel(columns_name[j],fontsize=18)
                else:
                    if j < fig_col-1:
                        if columns_label[j] == 'best_size':
                            axs[i,j].set_xticks([1,5]) # set x_tick for the best size
                            formatter = FuncFormatter(lambda x_val, tick_pos: "$2^{{{:.0f}}}$".format((x_val)))
                            axs[i,j].xaxis.set_major_formatter(formatter)
                        elif columns_label[j] == 'rf_size':
                            axs[i,j].set_xticks([2,3]) # set x_tick for the best size
                            formatter = FuncFormatter(lambda x_val, tick_pos: "$10^{{{:.0f}}}$".format((x_val)))
                            axs[i,j].xaxis.set_major_formatter(formatter)
                        else:
                            formatter = FuncFormatter(lambda x_val, tick_pos: "{:g}".format((x_val)))
                            axs[i,j].xaxis.set_major_formatter(formatter)
                        
                        axs[i,j].tick_params(axis='both', which='major', labelsize=18)
                        axs[i,j].set_yticks([])
                        axs[i,j].set_xlabel(columns_name[j],fontsize=18)
                    else:
                       axs[i,j].set_xlabel('FAC',fontsize=18)
                       axs[i,j].set_yticks([])
                       axs[i,j].tick_params(axis='both', which='major', labelsize=18)
                       axs[i,j].set_ylabel('Rel freq (%)',fontsize=18)
            # if j==fig_col and i == fig_row-1:
            #     axs[i,j].set_ylabel('Relative frequency')
            
            # sns.distplot(df_results.iloc[:,j],ax=axs[i,j], 
            #              bins = bins_list[j], norm_hist=True, kde=False,hist_kws={"color":"blue"})
            # sns.distplot(df_results.loc[df_results['genetic_label_num']==i].iloc[:,j],ax=axs[i,j], 
            #              bins = bins_list[j], norm_hist=True, kde=False,hist_kws={"color":"red"})
    if len(file_save)>0:
        plt.savefig(file_save+'.png',bbox_inches='tight')
        plt.savefig(file_save+'.svg',bbox_inches='tight')
        plt.savefig(file_save+'.pdf',bbox_inches='tight')
    return freq_fac_all, np.asarray(freq_fac), data_box_clusters
#%% example: traces with error bars, barplot of single-trial amplitude for looming, polar graph, RF
def figure_example_temporal_bar_polor_rf(temporal_list, temporal_zooms,
                                 dl_amp, mb_amp, rf_amp_all, rf_stim_center, fig_params, 
                                 file_save, sampling_rate=5):
    ''' example plots of traces with error bars, 
    barplot of single-trial amplitude for looming, polar graph, RF'''
    num_neuron,_ = dl_amp.shape
    fig_row = num_neuron
    fig_col_temporal = len(temporal_list)
    fig_col_other = 4 #barplot of single-trial amplitude for looming, polar graph, RF_on and RF_off

    fig_col = fig_col_temporal + fig_col_other
    fig_col_temporal_size = []
    for i in range(fig_col_temporal):
        _,x_len = temporal_list[i][1].shape
        fig_col_temporal_size.append(x_len*temporal_zooms[i])
        
    fig_col_other_size = [16,8,8,5] #
    
    # single histogram should have w:h = 16:10
    fig_height_au = num_neuron*(fig_params['height_sub'] + fig_params['space_sub'])
    fig_width_au = np.sum(fig_col_temporal_size) \
                    + np.sum(fig_col_other_size) \
                    + (fig_col_temporal+fig_col_other)*fig_params['space_sub']  
    # define the left and bottom position and width and height for all subplots
    # relative to figure width and height
    
    subplots_bottom = fig_params['margin'] + (1-2*fig_params['margin'])/fig_row*np.asarray(range(fig_row))
    subplots_width = 1-2*fig_params['margin']
    subplots_left_1 = subplots_width/fig_width_au*(np.cumsum(np.asarray(fig_col_temporal_size)+fig_params['space_sub']))
    subplots_left_2 = subplots_left_1[-1] + subplots_width/fig_width_au*(np.cumsum(np.asarray(fig_col_other_size)+fig_params['space_sub']))
    subplots_left = fig_params['offset'] + fig_params['margin'] + np.concatenate((np.asarray([0]),subplots_left_1,subplots_left_2[1:]), axis=0)
    # print(subplots_left)
    subplots_height = (1-2*fig_params['margin'])*fig_params['height_sub']/fig_height_au
    subplots_width_array = subplots_left[1:] - subplots_left[:-1] - subplots_width/fig_width_au*fig_params['space_sub'] 
    #add the width for the last column
    subplots_width_array = np.append(subplots_width_array,subplots_width/fig_width_au*fig_col_other_size[-1])
    
    fig_width_fin = fig_width*fig_params['zoom']
    fig_height_fin = fig_width_fin*fig_height_au/(2*fig_width_au)    
    fig = plt.figure(figsize=[fig_width_fin,fig_height_fin])        
    
    # set the y_limits for temporal plot
    y_limit_list = []
    for i in range(fig_row):
        y_limit_j = []
        for j in range(fig_col_temporal):          
            y_limit_j.append(temporal_list[j][1][i,:].max() - temporal_list[j][1][i,:].min())
        y_limit_list.append(max(y_limit_j))
    y_limit_all = max(y_limit_list)
    
    k = 99
    rf_stim = (11,11)
    rad = np.asarray(range(13))*np.pi/6
    # for i in range(2):
    #     for j in range(fig_col):
    for i in range(fig_row):
        for j in range(fig_col):
            # print(i)
            # print(j)
            bottom = subplots_bottom[i]
            left = subplots_left[j]
            width = subplots_width_array[j]
            height = subplots_height
            if j<fig_col_temporal:
                ax = fig.add_axes([left,bottom,width,height])
                y = temporal_list[j][1][i,:]
                ye = temporal_list[j][2][i,:]
                t = np.asarray(range(1, y.shape[0]+1))/sampling_rate
                ax.plot(t,y, color='black')   
                ax.fill_between(t, y-ye, y+ye, color='grey')
                ax.set_ylim([y.min()-y_limit_list[i]*0.1, y.min()+y_limit_list[i]*1.1])       
                # if j==0 and i == 0:
                #     ax.set_ylabel('Normalized amplitude')                      
            else:
                k = j - fig_col_temporal
                # barplots for looming
                if k == 0:  
                    ax = fig.add_axes([left,bottom,width,height])
                    # ax.bar(np.asarray(range(10))+1,dl_amp[i,:], width=1, align='center', edgecolor='black',color='green', alpha=1)
                    ax.plot(np.asarray(range(10))+1,dl_amp[i,:], marker='o',c='k',markersize=8)
                    ax.set_xticks([]) 
                    ax.set_yticks([]) 
                    ax.spines['right'].set_visible(False)
                    ax.spines['top'].set_visible(False)
                    y_max = dl_amp[i,:].max()
                    y_min = dl_amp[i,:].min()
                    y_range= y_max-y_min
                    # ax.set_ylim([y_min-y_range*0.1, y_max+y_range*0.1])   
                    ax.set_ylim([0, y_max+y_range*0.1])   
                if k == 1:
                    ax = fig.add_axes([left,bottom,width,height], projection='polar')
                    ax.plot(rad, np.append(mb_amp[i,:],mb_amp[i,0]), color='black')
                    ax.set_theta_direction(-1)
                    # ax.set_theta_zero_location('N')
                if k == 2:
                    # plot both on and off receptive fields
                    rf_min = rf_amp_all[i,:].min()
                    rf_max = rf_amp_all[i,:].max()
                    ax = fig.add_axes([left,bottom,width,height])
                    ax.imshow(np.reshape(rf_amp_all[i,:np.prod(rf_stim)],rf_stim), vmin=rf_min, vmax=rf_max,cmap='viridis')
                if k == 3:
                    ax = fig.add_axes([left,bottom,width,height])
                    ax.imshow(np.reshape(rf_amp_all[i,np.prod(rf_stim):],rf_stim), vmin=rf_min, vmax=rf_max,cmap='viridis')
            if i > 0:
                # ax.axes.xaxis.set_ticklabels([])
                # ax.axes.yaxis.set_ticklabels([])
                if ax.name != 'polar':
                    ax.spines['right'].set_visible(False)
                    ax.spines['top'].set_visible(False)
                    ax.set_xticks([]) 
                    ax.set_yticks([])
                else:
                    ax.tick_params(labelbottom=False) 
                    ax.tick_params(labelleft=False) 
                # ax.axis("off")
            else:
                # print(ax.name)
                if ax.name != 'polar':
                    ax.spines['right'].set_visible(False)
                    ax.spines['top'].set_visible(False)
                            # Only show ticks on the left and bottom spines
                    ax.yaxis.set_ticks_position('left')
                    ax.xaxis.set_ticks_position('bottom')
                    ax.set_xticks([]) 
                    ax.set_yticks([])
                else:
                    ax.tick_params(labelbottom=False) 
                    ax.tick_params(labelleft=False) 
                    # ax.set_xticks([]) 
                    # ax.set_yticks([])

    if len(file_save)>0:
        plt.savefig(file_save+'.png',bbox_inches='tight')
        plt.savefig(file_save+'.svg',bbox_inches='tight')
        plt.savefig(file_save+'.pdf',bbox_inches='tight')
        
    return -1

#%%
def density_recovery_profile(c,bin_size,w,h):
    '''plot density recovery profiles for c; c is a n x 2 numpy array'''
    # remove nan from the array
    c = c[~np.isnan(c).any(axis=1),:]
    if w==0 and h==0:
        w = c[:,0].max() - c[:,0].min()
        h = c[:,1].max() - c[:,1].min()
    # print(c.shape)
    n,_ = c.shape
    # calculate N*dr: average number of cells at radii between r and r+dr
    # calculate distance between all pairs
    dist = np.zeros((n,n))
    for i in range(n):
        for j in range(i,n):
            dist[i,j] = distance.euclidean(c[i,:], c[j,:])
            dist[j,i] = dist[i,j]
    for i in range(n):  
        dist[i,i] = np.nan
    # histograms the distance
    dist_max = np.nanmax(dist)
    #if using analytic approach, dist_max should be the maximal distance within the region of interest
    dist_max = np.sqrt(w**2+h**2)*1.05
    # print(bin_size)
    bins = np.arange(0, dist_max, bin_size)
    bin_num = bins.size-1
    hist = np.ones((n,bin_num))*np.nan
    for i in range(n):
        hist[i,:],_ = np.histogram(dist[i,:],bins=bins)
    hist_mean = np.nanmean(hist,axis=0)
    nn = hist_mean/bin_size # this is N in zhang 2012  
    # plt.plot(bins[1:],nn,'.')
    
    # #compute the average area available at radii between r and r+dr from a reference cell
    # # analytical approach for square as in zhang 2012
    # # r_max = bins.max()
    # a = c.max() - c.min() # edge length of the square
    # aa = np.ones(bin_num)*np.nan
    # _idx = bisect.bisect_left(bins, a)
    # # when r<=a
    # r = (bins[:_idx]+bins[1:_idx+1])/2
    # z = r/a
    # aa[:_idx] = 2*r*(np.pi-4*z+z**2)
    # # when r>=a
    # r = (bins[_idx:-1]+bins[_idx+1:])/2
    # z = r/a
    # aa[_idx:] = 2*r*(4*np.arcsin(1/z)-np.pi-2+4*np.sqrt(z**2-1)-z**2)
    
    # analytical approach for rectangle, w>h, w is along x axis, h is long y axis 
    # w = c[:,0].max() - c[:,0].min()
    # h = c[:,1].max() - c[:,1].min()
    # rotate rectangle if w<h
    cc = np.zeros_like(c)
    if w>=h:
        cc = c
    else:
        cc[:,0] = c[:,1]
        cc[:,1] = c[:,0]
        w,h = h,w
    aa = np.ones(bin_num)*np.nan
    _idx_h = bisect.bisect_left(bins, h)
    r = (bins[:_idx_h]+bins[1:_idx_h+1])/2
    aa[:_idx_h] = 2*r*(np.pi-2*r*((1/h)+(1/w))+((r**2)/(h*w)))
    _idx_w = bisect.bisect_left(bins, w)
    # if _idx_w<bins.shape[0]:
    r = (bins[_idx_h:_idx_w]+bins[_idx_h+1:_idx_w+1])/2
    aa[_idx_h:_idx_w] = 2*r*(2*np.arcsin(h/r) - 2*r/h + 2*np.sqrt((r/h)**2-1)-h/w)
    r = (bins[_idx_w:-1]+bins[_idx_w+1:])/2
    aa[_idx_w:] = 2*r*(2*np.arcsin(h/r) + 2*np.arcsin(w/r) - np.pi - h/w - w/h 
                       + 2*np.sqrt((r/h)**2-1) + 2*np.sqrt((r/w)**2-1) - ((r**2)/(h*w)))
    
    
    
    
    density = nn/aa
    plt.bar(bins[1:],density,bin_size,edgecolor='black')
    plt.show()
    # plt.plot(bins[1:],nn)
    # plt.plot(bins[1:],aa*253)
    # plt.show()
    
    
    # # dist_min = dist[dist!=0].min()        
    # # dist_max = dist.max()
    # # bins = np.linspace(dist_min, dist_max, num=bin_num)

    # # density_all = np.zeros((n,bin_num))
    # # for i in range(n):
    # #     for j in range(bin_num):
    # #         density_all[i,j] = np.sum(dist[i,:]<=bins[j])/(np.pi*bins[j]**2)
    # # density = np.mean(density_all,axis=0)
    # # an alternative way is set the refence as the neuron in the center of the image
    # c_mean = np.nanmean(c,axis=0)
    # dist_min_temp = 1e99
    # idx_center = 0
    # for i in range(n):
    #     if not np.isnan(c[i,:]).any():
    #         dist_temp = distance.euclidean(c[i,:], c_mean)
    #         if dist_temp<dist_min_temp:
    #             dist_min_temp = dist_temp
    #             idx_center = i
    # dist = np.ones(n)*np.nan
    # for i in range(n):
    #     if not np.isnan(c[i,:]).any():
    #         dist[i] = distance.euclidean(c[i,:], c[idx_center,:])
    # # dist_min = dist[dist!=0].min()        
    # # dist_max = dist.max()
    # # bins = np.linspace(dist_min, dist_max, num=bin_num)
    # bin_num = len(bins)
    # density = np.zeros(bin_num)
    # for i in range(bin_num):
    #     density[i] = np.sum(dist<=bins[i])/(np.pi*bins[i]**2)
    return density, bins, nn, aa

# Histogram pairwise distances
def Histo_Dist(pts,dr):
    '''
    Histograms all pairwise distances among the points in pts, with bin size dr.
    Note there are a total of n*(n-1)/2 values with n=len(pts)
    '''
    di=np.array([pts[i]-pts[j] for i in range(len(pts)-1) for j in range(i+1,len(pts))]) # all pairwise differences
    d=np.linalg.norm(di,axis=-1) # all pairwise distances
    m=int(np.ceil(np.max(d)/dr)) # number of distance bins + 1
    bins=dr*np.arange(m) # bin boundaries
    r=(bins[:-1]+bins[1:])/2 # bin centers
    N=np.histogram(d,bins=bins)[0] # number of pairwise distances per reference cell in each radius bin
    N=N/(len(pts)/2)/dr # normalize to one reference cell and per unit radius
    return r,N
# The area density for a rectangular window
def Area_Dist(r,w,h):
    '''
    Computes d(Area)/d(Radius) as a function of Radius from an average point in 
    a rectangular window of dimensions w x h.
    '''
    if w<h: # if width < height swap the two
        w,h=h,w
    if r<h:
        q=np.pi-2*r*(1/h+1/w)+r**2/h/w
    elif r<w:
        q=2*np.arcsin(h/r)-2*r/h+2*np.sqrt((r/h)**2-1)-h/w
    else:
        q=(2*np.arcsin(h/r)+2*np.arcsin(w/r)-np.pi-h/w-w/h+
           2*np.sqrt((r/h)**2-1)+2*np.sqrt((r/w)**2-1)-r**2/h/w)
    return 2*r*q
# Analyze pairwise distances on a rectangular window
def Analyze_Dist(pts,dr,w,h):
    '''
    Analyzes pairwise distances among the points in pts, with bin size dr,
    observed in a rectangular window of dimensions w x h.
    '''
    r,N=Histo_Dist(pts,dr)
    A=np.array([Area_Dist(x,w,h) for x in r])
    D=N/A
    E=np.sqrt(N/(len(pts)/2)/dr)/A
    return r,D,E,N,A # radius, density, uncertainty

def SortVertice(p):
    # compute centroid
    cent=(sum([x[0] for x in p])/len(p),sum([x[1] for x in p])/len(p))
    # sort by polar angle
    ps = np.asarray(sorted(p,key=lambda x: math.atan2(x[1]-cent[1],x[0]-cent[0])))
    return ps

def PolyArea(p):
    '''
    returns the area of a polygon defined by vertices in n x 2 array p
    see "Shoelace formula"
    '''
    return 0.5*np.abs(np.dot(p[:,0],np.roll(p[:,1],1))-np.dot(p[:,1],np.roll(p[:,0],1)))

def Sample_Poly(p,n):
    '''
    returns n points drawn uniformly from within the polygon p
    '''
    from matplotlib import path
    pa=path.Path(p) # convert polygon to path 
    rl,rr,rb,rt=min(p[:,0]),max(p[:,0]),min(p[:,1]),max(p[:,1]) # enclosing rectangle
    w,h=rr-rl,rt-rb # width and height of enclosing rectangle
    a=PolyArea(p) # area of the polygon
    na=int((n+10*np.sqrt(n))*h*w/a) # approx number of random points needed in the enclosing rectangle
    x=np.random.uniform(low=rl,high=rr,size=na) # pick random x values
    y=np.random.uniform(low=rb,high=rt,size=na) # pick random y values
    pts=np.stack([x,y],axis=-1) # convert to n x 2 array of points
    pts=pts[pa.contains_points(pts)] # points inside polygon
    return pts[:n] # first n of these

def Pdist_Poly(p,n,dr):
    '''
    Estimates a distance density on the polygon defined by vertices in the n x 2 array p
    uses n random points
    samples the distance density at a radius resolution of dr
    '''
    pts=Sample_Poly(p,n)
    r,N=Histo_Dist(pts,dr) # histogram pairwise differences
    N/=(np.sum(N)*dr) # normalize to an integral of 1
    return r,N
# Analyze pairwise distances on a polygonal window
def Analyze_Dist_Poly(pts,dr,p):
    '''
    Analyzes pairwise distances among the points in pts, with bin size dr,
    observed in a window given by the polygon with vertices in p[]
    '''
    rn,N=Histo_Dist(pts,dr)
    n=1000 # number of points to estimate distance density
    ra,A=Pdist_Poly(p,n,dr)
    A*=PolyArea(p)
    m=min(len(ra),len(rn)) # truncate everything to shorter series
    r=ra[:m];N=N[:m];A=A[:m] # truncate everything to shorter series
    D=N/A
    E=np.sqrt(N/(len(pts)/2)/dr)/A
    return r,D,E # radius, density, uncertainty




def jaccard(y_true, y_pred):
    """Calculate Jaccard coefficient

    Ranges from 0 to 1. Higher values indicate greater similarity.

    Parameters
    ----------
    y_true : list or array
        Boolean indication of cluster membership (1 if belonging, 0 if
        not belonging) for actual labels
    y_pred : list or array
        Boolean indication of cluster membership (1 if belonging, 0 if
        not belonging) for predicted labels

    Returns
    -------
    float
        Jaccard coefficient value
    """
    if type(y_true) is list:
        y_true = np.array(y_true)

    if type(y_pred) is list:
        y_pred = np.array(y_pred)
    # return float(np.sum(y_true & y_pred)) / (np.sum(y_true) + np.sum(y_pred) - np.sum(y_true & y_pred))
    return float(np.sum(np.logical_and(y_true,y_pred))) / float(np.sum(np.logical_or(y_true,y_pred)))

#%%
def seriation(Z,N,cur_index):
    '''
        input:
            - Z is a hierarchical tree (dendrogram)
            - N is the number of points given to the clustering process
            - cur_index is the position in the tree for the recursive traversal
        output:
            - order implied by the hierarchical tree Z
            
        seriation computes the order implied by a hierarchical tree (dendrogram)
    '''
    if cur_index < N:
        return [cur_index]
    else:
        left = int(Z[cur_index-N,0])
        right = int(Z[cur_index-N,1])
        return (seriation(Z,N,left) + seriation(Z,N,right))
    
def compute_serial_matrix(dist_mat,method="ward"):
    '''
        input:
            - dist_mat is a distance matrix
            - method = ["ward","single","average","complete"]
        output:
            - seriated_dist is the input dist_mat,
              but with re-ordered rows and columns
              according to the seriation, i.e. the
              order implied by the hierarchical tree
            - res_order is the order implied by
              the hierarhical tree
            - res_linkage is the hierarhical tree (dendrogram)
        
        compute_serial_matrix transforms a distance matrix into 
        a sorted distance matrix according to the order implied 
        by the hierarchical tree (dendrogram)
    '''
    N = len(dist_mat)
    flat_dist_mat = squareform(dist_mat)
    # res_linkage = linkage(flat_dist_mat, method=method,preserve_input=True) # this is for fastcluster
    res_linkage = linkage(flat_dist_mat, method=method)
    res_order = seriation(res_linkage, N, N + N-2)
    seriated_dist = np.zeros((N,N))
    a,b = np.triu_indices(N,k=1)
    seriated_dist[a,b] = dist_mat[ [res_order[i] for i in a], [res_order[j] for j in b]]
    seriated_dist[b,a] = seriated_dist[a,b]
    
    return seriated_dist, res_order, res_linkage
# %% sort X according to cluster
def sort_to_cluster(X_data,labels_gmm):
    '''
    sort X according to cluster

    Parameters
    ----------
    X_data : TYPE
        DESCRIPTION.
    labels_gmm : TYPE
        DESCRIPTION.

    Returns
    -------
    X_sorted : TYPE
        DESCRIPTION.

    '''
    n_cluster = np.unique(labels_gmm).size
    cluster_center = np.zeros((n_cluster,X_data.shape[1]))
    for i in range(n_cluster):
        cluster_center[i,:] = np.mean(X_data[labels_gmm==i,:],axis=0)
    
    cluster_center_min = np.min(cluster_center,axis=0)
    cluster_center_max = np.max(cluster_center,axis=0)
    cluster_center_range = cluster_center_max - cluster_center_min
    cluster_center_dim_max_range = np.argmax(cluster_center_range)
    plt.plot(cluster_center[:,0],cluster_center[:,1],'-^',color='black')
    plt.show()
    cluster_center_sorted = np.zeros_like(cluster_center)
    cluster_center_sorted_idx = np.zeros(n_cluster,dtype=int)
    cluster_center_sorted_idx[0] = np.argmin(cluster_center[:,cluster_center_dim_max_range])
    cluster_center_sorted[0,:] = cluster_center[cluster_center_sorted_idx[0],:]
    idx_left = np.ones(n_cluster,dtype=int)
    idx_left[cluster_center_sorted_idx[0]]=0
    for i in range(1,n_cluster):
        _temp = np.linalg.norm(cluster_center-cluster_center_sorted[i-1,:],axis=1)
        _idx_temp = np.argsort(_temp)
        cluster_center_sorted_idx[i] = _idx_temp[idx_left[_idx_temp]>0][0]
        cluster_center_sorted[i,:] = cluster_center[cluster_center_sorted_idx[i],:]
        idx_left[cluster_center_sorted_idx[i]] = 0
    plt.plot(cluster_center_sorted[:,0],cluster_center_sorted[:,1],'-^',color='black')
    plt.show()
    
    X_sorted = np.zeros_like(X_data)
    j=0
    for i in range(n_cluster):
        _temp = X_data[labels_gmm==cluster_center_sorted_idx[i],:]
        plt.plot(_temp[:,0],_temp[:,1],'-o')
        X_sorted[j:j+_temp.shape[0]] = _temp
        j = j+_temp.shape[0]
    plt.plot(cluster_center[:,0],cluster_center[:,1],'^',color='black')
    plt.show()
    plt.plot(X_sorted[:,0],X_sorted[:,1],'-^',color='black')
    plt.show()
    
    return X_sorted

#%% create co-associate matrix using n_cluster: 10, 15, 20, 25, 30, 35
def cal_co_mat(X,n_cluster_arr,random_states_arr):
    num_neuron = X.shape[0]
    cluster_paras = np.reshape(np.asarray(np.meshgrid(n_cluster_arr,random_states_arr)),(2,-1))
    n_co_mat = n_cluster_arr.size*(random_states_arr.size*2+1)
    co_mat = np.zeros((n_co_mat,num_neuron,num_neuron))
    for i in range(n_co_mat):
        print(i)
    # for i in range(1):
        #for gmm
        if i<cluster_paras.shape[1]: 
            n_cluster = cluster_paras[0,i]
            gmm = GaussianMixture(n_components=n_cluster,covariance_type='diag',random_state = cluster_paras[1,i])
            gmm.fit(X)
            labels = gmm.predict(X)
        #for spectral
        elif i>=cluster_paras.shape[1] and i<cluster_paras.shape[1]*2: 
            n_cluster = cluster_paras[0,i-cluster_paras.shape[1]]
            spectral = cluster.SpectralClustering(n_clusters=n_cluster, random_state = cluster_paras[1,i-cluster_paras.shape[1]],
                                              eigen_solver='arpack', affinity="nearest_neighbors")
            spectral = spectral.fit(X)
            labels = spectral.labels_
        else:
            n_cluster = n_cluster_arr[i-cluster_paras.shape[1]*2]
            ahc = AgglomerativeClustering(distance_threshold=None, n_clusters=n_cluster,affinity='euclidean',linkage='ward')
            ahc = ahc.fit(X)
            labels = ahc.labels_
        start = timeit.default_timer()
        co_mat[i,:,:] = np.ones((num_neuron,num_neuron))
        for j in range(num_neuron):
            # print(j)
            for k in range(j+1,num_neuron):
                if labels[j]!=labels[k]:
                    co_mat[i,j,k] = 0
                    co_mat[i,k,j] = 0
        stop = timeit.default_timer()
        print(stop-start)
        
    co_mat_mean = np.mean(co_mat,axis=0)
    plt.imshow(co_mat_mean,cmap='viridis')  
    plt.colorbar()
    plt.show()
    return co_mat_mean,co_mat
def cal_co_mat_gmm_spectral(X,n_cluster,random_states_arr):
    num_neuron = X.shape[0]
    # cluster_paras = np.reshape(np.asarray(np.meshgrid(n_cluster_arr,random_states_arr)),(2,-1))
    n_co_mat = random_states_arr.size*2
    co_mat_sum = np.zeros((num_neuron,num_neuron))
    for i in range(n_co_mat):
        if np.mod(i,10)==0:
            print(i)
    # for i in range(1):
        #for gmm
        if i<random_states_arr.size: 
            gmm = GaussianMixture(n_components=n_cluster,covariance_type='diag',random_state = random_states_arr[i],max_iter=1000)
            gmm.fit(X)
            labels = gmm.predict(X)
        #for spectral
        else: 
            spectral = cluster.SpectralClustering(n_clusters=n_cluster, random_state = random_states_arr[i-random_states_arr.size],
                                              eigen_solver='arpack', affinity="nearest_neighbors")
            spectral = spectral.fit(X)
            labels = spectral.labels_
        # start = timeit.default_timer()
        co_mat = np.ones((num_neuron,num_neuron))
        for j in range(num_neuron):
            # print(j)
            for k in range(j+1,num_neuron):
                if labels[j]!=labels[k]:
                    co_mat[j,k] = 0
                    co_mat[k,j] = 0
        co_mat_sum = co_mat_sum+co_mat
        # stop = timeit.default_timer()
        # print(stop-start)
        
    co_mat_mean = co_mat_sum/n_co_mat
    # plt.imshow(co_mat_mean,cmap='viridis')  
    # plt.colorbar()
    # plt.show()
    return co_mat_mean
def cal_co_mat_gmm_1(X,n_cluster,n_init=100):
    num_neuron = X.shape[0]
    co_mat_sum = np.zeros((num_neuron,num_neuron))
    labels = np.zeros((num_neuron,n_init),dtype=np.dtype('uint8'))
    n_co_mat = n_init
    for i in range(n_init):
        gmm = GaussianMixture(n_components=n_cluster,covariance_type='diag',random_state=i,max_iter=1000)
        gmm.fit(X)
        labels[:,i] = gmm.predict(X)
        # start = timeit.default_timer()
        co_mat = np.ones((num_neuron,num_neuron))
        for j in range(num_neuron):
            # print(j)
            for k in range(j+1,num_neuron):
                if labels[j,i]!=labels[k,i]:
                    co_mat[j,k] = 0
                    co_mat[k,j] = 0
        co_mat_sum = co_mat_sum+co_mat
        # stop = timeit.default_timer()       
    co_mat_mean = co_mat_sum/n_co_mat
    plt.imshow(co_mat_mean,cmap='viridis')  
    plt.colorbar()
    plt.show()
    return co_mat_mean
def cal_co_mat_gmm(X,labels):
    num_neuron = X.shape[0]
    n_co_mat = labels.shape[1]
    co_mat_sum = np.zeros((num_neuron,num_neuron))
    for i in range(n_co_mat):
        # start = timeit.default_timer()
        co_mat = np.ones((num_neuron,num_neuron))
        for j in range(num_neuron):
            # print(j)
            for k in range(j+1,num_neuron):
                if labels[j,i]!=labels[k,i]:
                    co_mat[j,k] = 0
                    co_mat[k,j] = 0
        co_mat_sum = co_mat_sum+co_mat
        # stop = timeit.default_timer()
        # print(stop-start)        
    co_mat_mean = co_mat_sum/n_co_mat
    plt.imshow(co_mat_mean,cmap='viridis')  
    plt.colorbar()
    plt.show()
    return co_mat_mean
def sort_co_mat(X, co_mat_mean):  
    num_neuron = X.shape[0]
    co_mat_mean_sorted = np.zeros_like(co_mat_mean)
    co_mat_mean_sorted_idx = np.zeros(num_neuron,dtype=int)
    X_min = np.min(X,axis=0)
    X_max = np.max(X,axis=0)
    X_range = X_max - X_min
    dim_max_range = np.argmax(X_range)
    # set start point as the most left point along longest dimension
    co_mat_mean_sorted_idx[0] = np.argmin(X[:,dim_max_range])
    # co_mat_mean_sorted_idx[0] = np.argmax(X[:,dim_max_range])
    for i in range(num_neuron-1):
        for j in range(num_neuron):
            k = np.argsort(co_mat_mean[co_mat_mean_sorted_idx[i],:])[::-1][j]
            if np.sum(k==co_mat_mean_sorted_idx[:i+1])==0:
                co_mat_mean_sorted_idx[i+1] = k
                break 
    # plt.plot(X[co_mat_mean_sorted_idx,0],X[co_mat_mean_sorted_idx,1],'^-')
    for i in range(num_neuron):
        co_mat_mean_sorted[i,i:] = co_mat_mean[co_mat_mean_sorted_idx[i],co_mat_mean_sorted_idx[i:]] 
        co_mat_mean_sorted[i:,i] = co_mat_mean_sorted[i,i:]
    # plt.imshow(co_mat_mean_sorted,cmap='viridis') 
    # plt.colorbar()
    # plt.show()
    return co_mat_mean_sorted,co_mat_mean_sorted_idx      
# %% define a function to sort co_mat with clusters
def sort_co_mat_cluster(X_data,labels_gmm,co_mat,plot_bool=False):
    n_cluster = np.unique(labels_gmm).shape[0]
    # n_feature = X_data.shape[1]
    n_neuron = labels_gmm.size
    # sort cluster, this is abandoned 20210211
    # cluster_center = np.zeros((n_cluster,n_feature))
    # for i in range(n_cluster):
    #     cluster_center[i,:] = np.mean(X_data[labels_gmm==i,:],axis=0)

    # cluster_center_min = np.min(cluster_center,axis=0)
    # cluster_center_max = np.max(cluster_center,axis=0)
    # cluster_center_range = cluster_center_max - cluster_center_min
    # cluster_center_dim_max_range = np.argmax(cluster_center_range)
    # # plt.plot(cluster_center[:,0],cluster_center[:,1],'-^',color='black')
    # # plt.show()
    # cluster_center_sorted = np.zeros_like(cluster_center)
    # cluster_center_sorted_idx = np.zeros(n_cluster,dtype=int)
    # cluster_center_sorted_idx[0] = np.argmin(cluster_center[:,cluster_center_dim_max_range])
    # cluster_center_sorted[0,:] = cluster_center[cluster_center_sorted_idx[0],:]
    # idx_left = np.ones(n_cluster,dtype=int)
    # idx_left[cluster_center_sorted_idx[0]]=0
    # for i in range(1,n_cluster):
    #     _temp = np.linalg.norm(cluster_center-cluster_center_sorted[i-1,:],axis=1)
    #     _idx_temp = np.argsort(_temp)
    #     cluster_center_sorted_idx[i] = _idx_temp[idx_left[_idx_temp]>0][0]
    #     cluster_center_sorted[i,:] = cluster_center[cluster_center_sorted_idx[i],:]
    #     idx_left[cluster_center_sorted_idx[i]] = 0
    # plt.plot(cluster_center_sorted[:,0],cluster_center_sorted[:,1],'-^',color='black')
    # plt.show()
    co_mat_sorted_cluster = np.zeros_like(co_mat)
    neuron_sorted_idx = np.zeros(n_neuron,dtype=int)
    j = 0
    for i in range(n_cluster):
        _temp = np.where(labels_gmm==i)
        neuron_sorted_idx[j:j+_temp[0].shape[0]] = _temp[0]
        j = j+_temp[0].shape[0]
    # neuron_sorted_idx = np.asarray(neuron_sorted_idx_list,dtype=int)
    for i in range(n_neuron):
        co_mat_sorted_cluster[i,i:] = co_mat[neuron_sorted_idx[i],neuron_sorted_idx[i:]]
        co_mat_sorted_cluster[i:,i] = co_mat_sorted_cluster[i,i:]                                    
    if plot_bool:
        fig, axs = plt.subplots(1,1)
        axs.pcolormesh(co_mat_sorted_cluster[::-1,::-1])
        axs.set_aspect('equal', 'box')
        plt.show()
    return co_mat_sorted_cluster,neuron_sorted_idx
    
    # co_mat_mean_sorted = np.zeros_like(co_mat_mean)
    # co_mat_mean_sorted_idx = np.zeros(num_neuron,dtype=int)
    # X_min = np.min(X,axis=0)
    # X_max = np.max(X,axis=0)
    # X_range = X_max - X_min
    # dim_max_range = np.argmax(X_range)
    # # set start point as the most left point along longest dimension
    # co_mat_mean_sorted_idx[0] = np.argmin(X_data[:,dim_max_range])
    # for i in range(num_neuron-1):
    #     for j in range(num_neuron):
    #         k = np.argsort(co_mat_mean[co_mat_mean_sorted_idx[i],:])[::-1][j]
    #         if np.sum(k==co_mat_mean_sorted_idx[:i+1])==0:
    #             co_mat_mean_sorted_idx[i+1] = k
    #             break 
    # plt.plot(X[co_mat_mean_sorted_idx,0],X[co_mat_mean_sorted_idx,1],'^-')
    # for i in range(num_neuron):
    #     co_mat_mean_sorted[i,i:] = co_mat_mean[co_mat_mean_sorted_idx[i],co_mat_mean_sorted_idx[i:]] 
    #     co_mat_mean_sorted[i:,i] = co_mat_mean_sorted[i,i:]
    # fig, axs = plt.subplots(1,1)
    # axs.pcolormesh(co_mat_mean_sorted)
    # axs.set_aspect('equal', 'box')
    # plt.show()
#%% calculate within-cluster rate and between-cluster rate, merge clusters if there are close
def cal_co_cluster(co_mat_mean_s_sorted_cluster,cluster_size,plot_bool=True):
    n_cluster_optimal = cluster_size.shape[0]
    cluster_size_gmm_cumsum = np.cumsum(cluster_size).astype(int) #order reversed to match the plot
    co_cluster = np.zeros((n_cluster_optimal,n_cluster_optimal))
    for i in range(n_cluster_optimal):
        if i == 0: 
            i_ini = 0
        else:
            i_ini = cluster_size_gmm_cumsum[i-1]
        i_fin = cluster_size_gmm_cumsum[i]
        for j in range(i,n_cluster_optimal):
            if j == 0:
                j_ini = 0
            else:
                j_ini = cluster_size_gmm_cumsum[j-1]
            j_fin = cluster_size_gmm_cumsum[j]
            co_cluster[i,j] = np.mean(co_mat_mean_s_sorted_cluster[i_ini:i_fin,j_ini:j_fin])
            co_cluster[j,i] = co_cluster[i,j]
    if plot_bool:
        _, axs = plt.subplots(1,1)
        axs.pcolormesh(co_cluster[::-1,::-1],cmap='viridis')
        axs.set_aspect('equal', 'box')
        axs.set_xticks([])
        axs.set_yticks([])
        plt.show()   
    return co_cluster
# %% subsampling to test the cluster stability
def cal_sub_sampling(X,n_cluster_optimal,sub_perc=0.9,num_sub=20,n_init=100,random_init=True):
    print('subsampling...')
    num_neuron = X.shape[0]
    num_feature = X.shape[1]
    num_neuron_sub = int(np.round(num_neuron*sub_perc))
    X_sub = np.ones((num_neuron_sub,num_feature,num_sub))*np.nan
    X_cluster_gmm_sub = np.ones((n_cluster_optimal,num_feature,num_sub))*np.nan
    idx_sub = np.ones((num_neuron_sub,num_sub),dtype=int)*np.nan
    labels_gmm_sub = np.zeros((num_neuron_sub,num_sub),dtype=np.dtype('uint8'))
    if not random_init:
        labels_gmm_sub_arr = np.zeros((num_neuron_sub,num_sub,n_init),dtype=np.dtype('uint8'))
    random_seeds = np.asarray(range(num_sub))
    for i in range(num_sub):
        print(i)
        np.random.seed(random_seeds[i])
        idx_sub[:,i] = np.random.choice(num_neuron,num_neuron_sub,replace=False)
        X_sub[:,:,i] = X[idx_sub[:,i].astype(int),:]
        if random_init:
            gmm_sub = GaussianMixture(n_components=n_cluster_optimal,covariance_type='diag',n_init=n_init)
            gmm_sub.fit(X_sub[:,:,i])
            labels_gmm_sub[:,i] = gmm_sub.predict(X_sub[:,:,i])
        else:
            bic_arr = np.ones(n_init)*np.nan
            
            for j in range(n_init):
                gmm_sub = GaussianMixture(n_components=n_cluster_optimal,covariance_type='diag',
                                          random_state=j, max_iter=1000)
                gmm_sub.fit(X_sub[:,:,i])
                labels_gmm_sub_arr[:,i,j] = gmm_sub.predict(X_sub[:,:,i])
                bic_arr[j] = gmm_sub.bic(X_sub[:,:,i])
            # bic = np.min(bic_arr)
            bic_idx = np.argmin(bic_arr)
            labels_gmm_sub[:,i] = labels_gmm_sub_arr[:,i,bic_idx]
        # show clusters and their distance in feature space
        X_cluster_gmm_sub[:,:,i],_,_,_ = plot_cluster(X_sub[:,:,i],labels_gmm_sub[:,i],[],sampling_rate=1,plot_flag=False)
    return X_cluster_gmm_sub, labels_gmm_sub,idx_sub

#%% calculate the correlation between clusters in original dataset
def cal_corr_sub(X_cluster_gmm_dendro,X_cluster_gmm_sub,num_sub=20,plot_bool=False):
    n_cluster_optimal = X_cluster_gmm_dendro.shape[0]
    corr_original = np.ones((n_cluster_optimal,n_cluster_optimal))*np.nan
    corr_original_full = np.ones((n_cluster_optimal,n_cluster_optimal))
    for i in range(n_cluster_optimal):
        for j in range(i+1,n_cluster_optimal):
            corr_original[i,j],_ = scipy.stats.pearsonr(X_cluster_gmm_dendro[i,:], X_cluster_gmm_dendro[j,:])
            corr_original_full[i,j] = corr_original[i,j]
            corr_original_full[j,i] = corr_original_full[i,j]
    # print(np.nanmax(corr_original))
    if plot_bool:
        plt.imshow(corr_original,cmap='bwr',vmin=-1,vmax=1)     
        plt.show()   
        plt.hist(corr_original.ravel(),bins=20)
        plt.show()


    # calculate the correlation between original and surrogate dataset
    corr_sub = np.ones((num_sub,n_cluster_optimal,n_cluster_optimal))*np.nan
    for i in range(num_sub):
        for j in range(n_cluster_optimal):
            for k in range(n_cluster_optimal):
                corr_sub[i,j,k],_ = scipy.stats.pearsonr(X_cluster_gmm_dendro[j,:],X_cluster_gmm_sub[k,:,i])
    corr_sub_match = np.max(corr_sub,axis=2)    
    return corr_original, corr_original_full, corr_sub_match
#%% calculate the distance between clusters in original dataset
def cal_dist_sub(X_cluster_gmm_dendro,X_cluster_gmm_sub,num_sub=20,plot_bool=False):
    n_cluster_optimal = X_cluster_gmm_dendro.shape[0]
    dist_original = np.ones((n_cluster_optimal,n_cluster_optimal))*np.nan
    dist_original_full = np.zeros((n_cluster_optimal,n_cluster_optimal))
    for i in range(n_cluster_optimal):
        for j in range(i+1,n_cluster_optimal):
            dist_original[i,j] = distance.euclidean(X_cluster_gmm_dendro[i,:], X_cluster_gmm_dendro[j,:])
            dist_original_full[i,j] = dist_original[i,j]
            dist_original_full[j,i] = dist_original_full[i,j]
    # print(np.nanmin(dist_original))
    dist_original_full_nan = np.copy(dist_original_full)
    dist_original_full_nan[dist_original_full==0] = np.nan
    if plot_bool:   
        plt.imshow(dist_original,cmap='viridis')     
        plt.show()   
        plt.hist(dist_original.ravel(),bins=20)
        plt.show()


    # calculate the correlation between original and surrogate dataset
    dist_sub = np.ones((num_sub,n_cluster_optimal,n_cluster_optimal))*np.nan
    for i in range(num_sub):
        for j in range(n_cluster_optimal):
            for k in range(n_cluster_optimal):
                dist_sub[i,j,k] = distance.euclidean(X_cluster_gmm_dendro[j,:],X_cluster_gmm_sub[k,:,i])
    dist_sub_match = np.min(dist_sub,axis=2)  
    return dist_original_full, dist_original_full_nan, dist_sub_match
#%%
def cal_jaccard_sub(labels_gmm_dendro,labels_gmm_sub,idx_sub,num_sub=20):
    n_cluster_optimal = np.unique(labels_gmm_dendro).size
    jaccard_sub = np.ones((num_sub,n_cluster_optimal,n_cluster_optimal))*np.nan
    for i in range(num_sub):
        for j in range(n_cluster_optimal):
            for k in range(n_cluster_optimal):
                jaccard_sub[i,j,k] = sklearn.metrics.jaccard_score(labels_gmm_dendro[idx_sub[:,i].astype(int)]==j,labels_gmm_sub[:,i]==k)
    jaccard_sub_match = np.max(jaccard_sub,axis=2)    
    return jaccard_sub_match


# %%
#%% calculate the significance of functional properties between one cluster and the others
# calculate the mean difference between one cluster and the others
# todo: in the plot, log rf first then bar plot. another way is bar plot in log axis.
def si_corr(df_results,hist_list,labels,label_exclude=999,ref_list=[]):
    n_fp = len(hist_list)
    labels_unique = np.unique(labels)
    n_cluster = labels_unique.size
    if label_exclude != 999:
        n_cluster = n_cluster - len(label_exclude)
        labels_unique = np.delete(labels_unique,label_exclude)
    si_cluster = np.zeros((n_fp, n_cluster))
    p_cluster = np.zeros((n_fp, n_cluster))
    for i in range(n_fp):
        fun_data = df_results[hist_list[i]]
        for j in range(n_cluster):
            c1 = fun_data[labels==labels_unique[j]]
            c1 = c1[~np.isnan(c1)]
            labels_temp_bool = np.ones_like(labels)>0
            if label_exclude != 999:
                for k in label_exclude:
                    labels_temp_bool[labels==k]=False
            labels_temp_bool[labels==labels_unique[j]] = False
            if len(ref_list) == 0:
                c0 = fun_data[labels_temp_bool]
                c0 = c0[~np.isnan(c0)]
                si_cluster[i,j] = c1.mean() - c0.mean()
                _,p_cluster[i,j] = scipy.stats.ttest_ind(c1, c0, equal_var=False, alternative='two-sided')
            else:
                si_cluster[i,j] = c1.mean() - ref_list[i]
                if ref_list[i] == 0 or ref_list[i] > 1:
                    _,p_cluster[i,j] = scipy.stats.ttest_1samp(c1, ref_list[i], axis=0, nan_policy='omit', alternative='two-sided')
                else:
                    _,p_cluster[i,j] = scipy.stats.ttest_1samp(c1, ref_list[i], axis=0, nan_policy='omit', alternative='two-sided') #was greater before 20211208
                    
            # if i==0:
            #     si_cluster[i,j] = si_cluster[i,j]/100
            # si_cluster[i,j] = (c1.mean() - c0.mean())/(c1.mean() + c0.mean())
    
    
    # calculate how features are correlated across clusters, rf size is removed
    corr_fp = np.ones((n_fp,n_fp)) 
    corr_p_fp = np.zeros((n_fp,n_fp)) 
    for i in range(n_fp):
        for j in range(i+1,n_fp):
            corr_fp[i,j],corr_p_fp[i,j] = scipy.stats.pearsonr(si_cluster[i,:],si_cluster[j,:])
            corr_fp[j,i] = corr_fp[i,j]
            corr_p_fp[j,i] = corr_p_fp[i,j] 
    
    return si_cluster, p_cluster, corr_fp, corr_p_fp

def si_corr_feature(X,labels,ref,label_exclude=999):
    n_fp = X.shape[1]
    labels_unique = np.unique(labels)
    n_cluster = labels_unique.size
    if label_exclude != 999:
        n_cluster = n_cluster - len(label_exclude)
        labels_unique = np.delete(labels_unique,label_exclude)
    si_cluster = np.zeros((n_fp, n_cluster))
    p_cluster = np.zeros((n_fp, n_cluster))
    for i in range(n_fp):
        fun_data = X[:,i]
        for j in range(n_cluster):
            c1 = fun_data[labels==labels_unique[j]]
            c1 = c1[~np.isnan(c1)]
            labels_temp_bool = np.ones_like(labels)>0
            if label_exclude != 999:
                for k in label_exclude:
                    labels_temp_bool[labels==k]=False
            labels_temp_bool[labels==labels_unique[j]] = False
            if len(ref) == 0:
                c0 = fun_data[labels_temp_bool]
                c0 = c0[~np.isnan(c0)]
                si_cluster[i,j] = c1.mean() - c0.mean()
                _,p_cluster[i,j] = scipy.stats.ttest_ind(c1, c0, equal_var=False, alternative='two-sided')
            else:
                si_cluster[i,j] = c1.mean() - ref[i]
                if ref[i] == 0:
                    _,p_cluster[i,j] = scipy.stats.ttest_1samp(c1, ref[i], axis=0, nan_policy='omit', alternative='two-sided')
                    # _,p_cluster[i,j] = scipy.stats.ttest_1samp(c1, ref[i], axis=0, nan_policy='omit', alternative='greater')
                else:
                    _,p_cluster[i,j] = scipy.stats.ttest_1samp(c1, ref[i], axis=0, nan_policy='omit', alternative='greater')
                    
    
    
    # calculate how features are correlated across clusters, rf size is removed
    corr_fp = np.ones((n_fp,n_fp)) 
    corr_p_fp = np.zeros((n_fp,n_fp)) 
    for i in range(n_fp):
        for j in range(i+1,n_fp):
            corr_fp[i,j],corr_p_fp[i,j] = scipy.stats.pearsonr(si_cluster[i,:],si_cluster[j,:])
            corr_fp[j,i] = corr_fp[i,j]
            corr_p_fp[j,i] = corr_p_fp[i,j] 
    
    return si_cluster, p_cluster, corr_fp, corr_p_fp

def p_plot(p_val_true):
    p_val_plot = np.copy(p_val_true)
    p_val_plot[p_val_true>0.05] = 0
    p_val_plot[np.logical_and(p_val_true<0.05,p_val_true>0.01)] = 1
    p_val_plot[np.logical_and(p_val_true<0.01,p_val_true>0.001)] = 2
    p_val_plot[p_val_true<0.001] = 3
    return p_val_plot

def corr_genetic_fun(freq_fac_clusters_genetic,freq_thr=0):
    # freq_thr = 5 # only focus on clusters accournting >5%
    num_genetic = freq_fac_clusters_genetic.shape[0]
    corr_genetic_fac = np.ones((num_genetic,num_genetic)) 
    corr_p_genetic_fac = np.zeros((num_genetic,num_genetic)) 
    for i in range(num_genetic):
        for j in range(i+1,num_genetic):
            idx_temp = np.logical_or(freq_fac_clusters_genetic[i,:]>freq_thr,freq_fac_clusters_genetic[j,:]>=freq_thr)
            print(np.sum(idx_temp))
            corr_genetic_fac[i,j],corr_p_genetic_fac[i,j] = scipy.stats.pearsonr(freq_fac_clusters_genetic[i,idx_temp],
                                                                                 freq_fac_clusters_genetic[j,idx_temp])
            corr_genetic_fac[j,i] = corr_genetic_fac[i,j]
            corr_p_genetic_fac[j,i] = corr_p_genetic_fac[i,j] 
    corr_genetic_fac_sig = corr_genetic_fac*(corr_p_genetic_fac<0.05)
    _, axs = plt.subplots(1,1)
    axs.pcolormesh(corr_genetic_fac_sig)
    axs.set_aspect('equal', 'box')
    axs.set_xticks([])
    axs.set_yticks([])
    plt.show() 
    
    corr_p_genetic_fac_plot = p_plot(corr_p_genetic_fac)
    _, axs = plt.subplots(1,1)
    axs.pcolormesh(corr_p_genetic_fac_plot)
    axs.set_aspect('equal', 'box')
    axs.set_xticks([])
    axs.set_yticks([])
    plt.show() 

    return corr_genetic_fac, corr_p_genetic_fac


#%%
def sizeof_fmt(num, suffix='B'):
    ''' by Fred Cirera,  https://stackoverflow.com/a/1094933/1870254, modified'''
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f %s%s" % (num, 'Yi', suffix)

#%%
def project(v1,v2):
    return np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)) 

def sep_dist(x,y):    
    # dist = distance.cdist(x,y).mean() - (distance.pdist(x).mean() + distance.pdist(y).mean())
    dist = distance.euclidean(np.mean(x,0), np.mean(y,0))/(0.5*(distance.pdist(x).mean() + distance.pdist(y).mean()))
    return dist

def density_dist(dist_thr_arr_input,df_results,image,n_all_types):
    
    image_id_all = df_results['image_id']
    num_image = image['image_area'].size
    num_type = np.unique(df_results['cluster_label_dendro_num']).size
    num_neuron = df_results['cluster_label_dendro_num'].size
    neurons_per_image = image['neurons_per_image']
    
    idx_pair_sel = np.zeros((num_image,num_type,num_type))
    neurons_per_image = np.zeros(num_image)
    for i in range(num_image):
        neurons_per_image[i] = np.sum(df_results['image_id']==i)
    for i in range(num_image):
        for j in range(num_type):
            for k in range(j,num_type):
                if neurons_per_image[i]>50 and np.min(n_all_types[i,[k,j]])>=5:
                    idx_pair_sel[i,j,k] = 1
                    idx_pair_sel[i,k,j] = 1

    idx_pair_sel_sum = np.sum(idx_pair_sel,axis=0)
    idx_pair_sel_sum_reverse = idx_pair_sel_sum[::-1,::-1]
    idx_pair_sel_sum_reverse[idx_pair_sel_sum_reverse>0]=1
    
    if len(dist_thr_arr_input.shape)==1: #if it's 1D
        dist_thr_arr = np.broadcast_to(dist_thr_arr_input,(num_type,dist_thr_arr_input.size))
    else:
        dist_thr_arr = dist_thr_arr_input
    dist_num = dist_thr_arr.shape[1]    
    soma_position = np.stack((df_results['soma_pos_x'],df_results['soma_pos_y']),axis=1)
    labels_gmm_dendro = df_results['cluster_label_dendro_num']
    # the area of rings
    area_arr = np.zeros_like(dist_thr_arr)
    for di in range(dist_num):
        for i in range(num_type):
            if di == 0:
                area_arr[i,di] = np.pi*dist_thr_arr[i,di]**2
            else:
                area_arr[i,di] = np.pi*dist_thr_arr[i,di]**2 - np.pi*dist_thr_arr[i,di-1]**2
    # the cell density of each type in the whole image, #/mm^2
    density_whole_image = np.ones((num_image,num_type))*np.nan
    for i in range(num_image):
        if neurons_per_image[i]>50: 
            cell_type_img = labels_gmm_dendro[image_id_all==i]
            for j in np.unique(cell_type_img):
                density_whole_image[i,int(j)] = np.sum(cell_type_img==j)/(image['image_area'][i]/1e6)
    density_whole_image_mean = np.nanmean(density_whole_image,axis=0)
    density_whole_image_sd = np.nanstd(density_whole_image,axis=0)
    
    num_cell_type_nearby_sum_reverse_arr = np.ones((dist_num,num_type,num_type))*np.nan
    num_cell_type_nearby_mean_reverse_arr = np.ones((dist_num,num_type,num_type))*np.nan
    density_cell_type_nearby_sum_reverse_arr = np.ones((dist_num,num_type,num_type))*np.nan
    density_cell_type_nearby_mean_reverse_arr = np.ones((dist_num,num_type,num_type))*np.nan
    
    for di in range(dist_num):
        num_cell_type_nearby = np.ones((num_image,num_type,num_type))*np.nan
        density_cell_type_nearby = np.zeros((num_image,num_type,num_type))
        kkk = 0
        image_cnt = np.zeros(num_type)
        for i in range(num_image):
            if neurons_per_image[i]>50:
                num_cell_type_nearby_temp = np.zeros((num_type,num_type))
                # density_cell_type_nearby_temp = np.zeros((num_type,num_type))
                soma_position_img = soma_position[image_id_all==i,:]
                cell_type_img = labels_gmm_dendro[image_id_all==i]
                # choose cell types that have at least 10 neurons
                type_sel = np.where(np.sum(idx_pair_sel[i,:,:],axis=0)>0)[0]
                image_cnt[type_sel] = image_cnt[type_sel]+1
                soma_position_img_sel = np.empty(shape=(0, 2))
                cell_type_img_sel = np.empty(shape=(0),dtype=int)
                for t in type_sel:
                    soma_position_img_sel = np.concatenate((soma_position_img_sel,soma_position_img[cell_type_img==t,:]),axis=0)
                    cell_type_img_sel = np.concatenate((cell_type_img_sel,cell_type_img[cell_type_img==t]),axis=0)
                neuron_num_sel = cell_type_img_sel.size
                cell_num_nearby = np.zeros(neurons_per_image[i].astype(int))
                for j in range(neuron_num_sel):
                    dist_temp = np.linalg.norm(soma_position_img_sel[j,:]-soma_position_img_sel,axis=1)
                    idx_nearby = np.logical_and(dist_temp<dist_thr_arr[int(cell_type_img_sel[j]),di],dist_temp>0)
                    type_temp = cell_type_img_sel[idx_nearby]
                    cell_num_nearby[j] = np.sum(idx_nearby)
                    for kt in type_temp:
                        num_cell_type_nearby_temp[int(cell_type_img_sel[j]),int(kt)] = num_cell_type_nearby_temp[int(cell_type_img_sel[j]),int(kt)]+1
                        # density_cell_type_nearby_temp[cell_type_img_sel[j],kt] = 
                        kkk = kkk+1
                n_type_sel = np.zeros_like(type_sel)
                num_cell_type_nearby_temp_norm = np.zeros_like(num_cell_type_nearby_temp)
                for j,t in enumerate(type_sel):
                    n_type_sel[j] = np.sum(cell_type_img_sel==t)
                    num_cell_type_nearby_temp_norm[t,:] = num_cell_type_nearby_temp[t,:]/n_type_sel[j] 
                # this need to be divided by the number of neurons of that type
                num_cell_type_nearby[i,:,:] = num_cell_type_nearby_temp_norm

        for i in range(num_type):
            for j in range(num_image):
                if np.nansum(num_cell_type_nearby[j,:,i])==0:
                    num_cell_type_nearby[j,:,i] = np.nan
            # num_cell_type_nearby[num_cell_type_nearby==0] = np.nan
        num_cell_type_nearby_mean = np.nanmean(num_cell_type_nearby,axis=0)
        num_cell_type_nearby_sum = np.nansum(num_cell_type_nearby,axis=0)
        num_cell_type_nearby_sum_reverse = num_cell_type_nearby_sum[::-1,::-1]  
        num_cell_type_nearby_mean_reverse = num_cell_type_nearby_mean[::-1,::-1]
        # density_cell_type_nearby_sum_reverse = density_cell_type_nearby_sum[::-1,::-1]  
        num_cell_type_nearby_sum_reverse[idx_pair_sel_sum_reverse==0] = np.nan
        num_cell_type_nearby_sum_reverse_arr[di,:,:] = num_cell_type_nearby_sum_reverse
        num_cell_type_nearby_mean_reverse[idx_pair_sel_sum_reverse==0] = np.nan
        num_cell_type_nearby_mean_reverse_arr[di,:,:] = num_cell_type_nearby_mean_reverse
    # num_cell_type_nearby_sum_reverse_diff is the neuron number in a ring
        if di == 0:
           num_cell_type_nearby_sum_reverse_diff = np.copy(num_cell_type_nearby_sum_reverse) 
           num_cell_type_nearby_mean_reverse_diff = np.copy(num_cell_type_nearby_mean_reverse) 
        else:
           num_cell_type_nearby_sum_reverse_diff = num_cell_type_nearby_sum_reverse_arr[di,:,:] - num_cell_type_nearby_sum_reverse_arr[di-1,:,:]
           num_cell_type_nearby_mean_reverse_diff = num_cell_type_nearby_mean_reverse_arr[di,:,:] - num_cell_type_nearby_mean_reverse_arr[di-1,:,:]
        #%

        
        num_cell_type_nearby_sum_mean_reverse_diff = np.nanmean(num_cell_type_nearby_sum_reverse_diff,axis=1)
        # num_cell_type_nearby_sum_mean = np.mean(num_cell_type_nearby_sum,axis=1)
    
        density_cell_type_nearby_sum_reverse = np.zeros_like(num_cell_type_nearby_sum_reverse)
        density_cell_type_nearby_mean_reverse = np.zeros_like(num_cell_type_nearby_sum_reverse)
        for i in range(num_type):
        #     # normazlie the cell number by the average number of all types, this is abondaned on 20220118
            density_cell_type_nearby_sum_reverse[i,:] = num_cell_type_nearby_sum_reverse_diff[i,:]/num_cell_type_nearby_sum_mean_reverse_diff[i]

            density_cell_type_nearby_mean_reverse[i,:] = num_cell_type_nearby_mean_reverse_diff[i,:]/(density_whole_image_mean[i]*area_arr[i,di]/1e6)
    
    
        density_cell_type_nearby_sum_reverse[idx_pair_sel_sum_reverse==0] = np.nan
        density_cell_type_nearby_sum_reverse_arr[di,:,:] = density_cell_type_nearby_sum_reverse
        density_cell_type_nearby_mean_reverse[idx_pair_sel_sum_reverse==0] = np.nan
        density_cell_type_nearby_mean_reverse_arr[di,:,:] = density_cell_type_nearby_mean_reverse
    
        #%

    return density_cell_type_nearby_mean_reverse_arr, density_cell_type_nearby_sum_reverse_arr

def scatter_types(image_example_density,df_results,neuron,cluster_label_5_example_density_plot=[]):
    cluster_label_example_density = df_results['cluster_label_dendro_num'][neuron['image_id']==image_example_density]
    soma_position_example_density = neuron['soma_pos'][neuron['image_id']==image_example_density,:]
    num_type = np.unique(df_results['cluster_label_dendro_num']).size
    neurons_per_cluster_example_density = np.zeros(num_type)
    for i in range(num_type):
        neurons_per_cluster_example_density[i] = np.sum(cluster_label_example_density==i)
    if len(cluster_label_5_example_density_plot) == 0:
        # count cell number for each cluster
        # return to the first five clusters with maximal neurons
        cluster_label_5_example_density = neurons_per_cluster_example_density.argsort()[-5:][::-1]
        cluster_label_5_example_density_plot = num_type - cluster_label_5_example_density
    # cluster_label_5_example_density_plot = [7,24,14,21] #18, 24types
    else:
        cluster_label_5_example_density = num_type - cluster_label_5_example_density_plot
    soma_position_5_example_density = np.ones((int(neurons_per_cluster_example_density.max()),cluster_label_5_example_density.size*2))*np.nan
    w_example_density = soma_position_example_density[:,0].max() - soma_position_example_density[:,0].min()
    h_example_density = soma_position_example_density[:,1].max() - soma_position_example_density[:,1].min()
    for i in range(cluster_label_5_example_density.size):
        _temp = soma_position_example_density[cluster_label_example_density==cluster_label_5_example_density[i],:]
        soma_position_5_example_density[:_temp.shape[0],i*2:(i+1)*2] = _temp
    
    return soma_position_example_density,soma_position_5_example_density,cluster_label_5_example_density_plot



def rgc_sc_compare(rgc_dict,df_results,chirp_temporal_norm,color_temporal_norm,rnd_seed=0,sub_perc=1,plot_flag=False,bootstrap=False):
    sampling_rate = 5
    chirp_rgc = rgc_dict['chirp_rgc_avg_5hz']
    color_rgc = rgc_dict['color_rgc_avg_5hz']
    dsi_rgc = np.squeeze(rgc_dict['dsi_avg'])
    osi_rgc = np.squeeze(rgc_dict['osi_avg'])
    dsi_rgc_median = np.squeeze(rgc_dict['dsi_median'])
    osi_rgc_median = np.squeeze(rgc_dict['osi_median'])
    n_rgc = dsi_rgc.shape[0]
    type_num = np.unique(df_results['cluster_label_dendro_num']).size
    mb_gdsi = df_results['gdsi']
    mb_gosi = df_results['gosi']
    mb_gdsi_svd = df_results['gdsi_svd']
    mb_gosi_svd = df_results['gosi_svd']
    chirp_sc = np.zeros((chirp_temporal_norm.shape[1],type_num))
    color_sc = np.zeros((color_temporal_norm.shape[1],type_num))
    dsi_sc = np.zeros(type_num)
    osi_sc = np.zeros(type_num)
    dsi_sc_median = np.zeros(type_num)
    osi_sc_median = np.zeros(type_num)
    dsi_svd_sc = np.zeros(type_num)
    osi_svd_sc = np.zeros(type_num)
    r_sc = np.zeros(type_num)
    for i in range(type_num):
        _num_full = int(np.sum(df_results['cluster_label_dendro_num']==type_num-1-i))
        idx_sel = np.where(df_results['cluster_label_dendro_num']==type_num-1-i)[0]
        if not bootstrap:
            idx_temp = idx_sel.copy()
        else:          
            _num_sub = int(_num_full*sub_perc)
            np.random.seed(rnd_seed)
            idx_temp = idx_sel[np.random.choice(_num_full,_num_sub,replace=True)]
        chirp_sc[:,i] = np.mean(chirp_temporal_norm[idx_temp,:],axis=0)
        color_sc[:,i] = np.mean(color_temporal_norm[idx_temp,:],axis=0)
        dsi_sc[i] = np.mean(mb_gdsi[idx_temp])
        osi_sc[i] = np.mean(mb_gosi[idx_temp])
        dsi_sc_median[i] = np.median(mb_gdsi[idx_temp])
        osi_sc_median[i] = np.median(mb_gosi[idx_temp])
        dsi_svd_sc[i] = np.mean(mb_gdsi_svd[idx_temp])
        osi_svd_sc[i] = np.mean(mb_gosi_svd[idx_temp])
        r_sc[i] = np.corrcoef(mb_gdsi[idx_temp],mb_gosi[idx_temp])[0,1]

    # calculate the difference for dsi, define similarity as 1-difference
    dsi_rgc_norm = norm_x(dsi_rgc)
    dsi_sc_norm = norm_x(dsi_sc)
    dsi_diff_sc_rgc = np.zeros((type_num,n_rgc))
    for i in range(type_num):
        for j in range(n_rgc):
            dsi_diff_sc_rgc[i,j] = 1-np.abs(dsi_sc_norm[i] - dsi_rgc_norm[j])
            
    if plot_flag:
        plt.set_cmap('viridis')
        plt.imshow(dsi_diff_sc_rgc,aspect='equal')
        plt.title('similarity for dsi')
        plt.colorbar()
        plt.show()
    
    osi_rgc_norm = norm_x(osi_rgc)
    osi_sc_norm = norm_x(osi_sc)
    osi_diff_sc_rgc = np.zeros((type_num,n_rgc))
    for i in range(type_num):
        for j in range(n_rgc):
            osi_diff_sc_rgc[i,j] = 1-np.abs(osi_sc_norm[i] - osi_rgc_norm[j])
            
    if plot_flag:
        plt.set_cmap('viridis')
        plt.imshow(osi_diff_sc_rgc)
        plt.title('similarity for osi')
        plt.colorbar()
        plt.show()
    
    # calculate the correlation for chirp and color
    # chirp parameters in Badens: 32s = 2s(black)+3s(white)+3s(black)+2s(grey)+8s(freq)+2s(grey)+8s(amp)+2s(grey)+2s(black)
    # chirp parmaters in my stimuli: 37s = 3s(black)+3s(white)+3s(black)+3s(grey)+8s(freq)+3s(grey)+8s(amp)+3s(grey)+3s(black)
    # chirp parmaters in match: 30s = 3s(white)+3s(black)+2s(grey)+8s(freq)+2s(grey)+8s(amp)+2s(grey)+2s(black)
    
    chirp_sc_mean = np.mean(chirp_sc,axis=1)
    chirp_rgc_mean = np.mean(chirp_rgc,axis=1)
    if plot_flag:
        plt.plot(chirp_sc_mean)
        plt.plot(chirp_rgc_mean)
        plt.show()
    chirp_sc_match = np.concatenate((chirp_sc[round(3*sampling_rate)-1:round(11*sampling_rate)-1,:],
                                     chirp_sc[round(12*sampling_rate)-1:round(22*sampling_rate)-1,:],
                                     chirp_sc[round(23*sampling_rate)-1:round(33*sampling_rate)-1,:],
                                     chirp_sc[round(34*sampling_rate)-1:round(36*sampling_rate)-1,:]),axis=0)
    chirp_rgc_match = chirp_rgc[round(2*sampling_rate):,:]
    chirp_sc_match_mean = np.mean(chirp_sc_match,axis=1)
    chirp_rgc_match_mean = np.mean(chirp_rgc_match,axis=1)
    
    chirp_corr_sc_rgc = np.zeros((type_num,n_rgc))
    for i in range(type_num):
        for j in range(n_rgc):
            chirp_corr_sc_rgc[i,j] = np.corrcoef(chirp_sc_match[:,i],chirp_rgc_match[:,j])[0,1]
            
    if plot_flag:
        plt.set_cmap('bwr')
        plt.imshow(chirp_corr_sc_rgc,vmin=-1,vmax=1)
        plt.title('correlation for chirp')
        plt.colorbar()
        plt.show()
    
    
    # % color baden: 12s = 1s(black)+3s(green)+3s(black)+3s(blue)+2s(black)
    # color in my stimuli: 14s = (grey) + 1s(black)+3s(blue)+3s(black)+1s(black)+3s(green)+3s(black)
    # color parmaters in match: 12s = 1s(black)+3s(green)+3s(black)+3s(blue)+2s(black)
    color_sc_mean = np.mean(color_sc,axis=1)
    color_rgc_mean = np.mean(color_rgc,axis=1)
    if plot_flag:
        plt.plot(color_sc_mean,color='black')
        plt.plot(color_rgc_mean,color='blue')
        plt.show()
    color_sc_match = np.concatenate((color_sc[round(7*sampling_rate)-1:,:],
                                     color_sc[round(1*sampling_rate):round(6*sampling_rate)-1,:],),axis=0)
    color_rgc_match = color_rgc[:-2,:]
    color_sc_match_mean = np.mean(color_sc_match,axis=1)
    color_rgc_match_mean = np.mean(color_rgc_match,axis=1)
    
    color_corr_sc_rgc = np.zeros((type_num,n_rgc))
    for i in range(type_num):
        for j in range(n_rgc):
            color_corr_sc_rgc[i,j] = np.corrcoef(color_sc_match[:,i],color_rgc_match[:,j])[0,1]
    
    if plot_flag:
        plt.set_cmap('bwr')
        plt.imshow(color_corr_sc_rgc,vmin=-1,vmax=1)
        plt.title('correlation for color')
        plt.colorbar()
        plt.show()
    
    
    # find the best linear combination of RGCs to one type of SC neurons
    chirp_color_sc_match = np.concatenate((chirp_sc_match,color_sc_match),axis=0)
    chirp_color_rgc_match = np.concatenate((chirp_rgc_match,color_rgc_match),axis=0)
    # chirp_color_sc = norm_x(chirp_color_sc_match.T,formula=1).T
    # chirp_color_rgc = norm_x(chirp_color_rgc_match.T,formula=1).T
    chirp_color_sc_temp = (chirp_color_sc_match-chirp_color_sc_match.min())
    chirp_color_sc = chirp_color_sc_temp/chirp_color_sc_temp.max()
    chirp_color_rgc_temp = (chirp_color_rgc_match-chirp_color_rgc_match.min())
    chirp_color_rgc = chirp_color_rgc_temp/chirp_color_rgc_temp.max()

    dsi_osi_dict={'dsi_rgc_mean': dsi_rgc,'dsi_rgc_median': dsi_rgc_median,
                  'osi_rgc_mean': osi_rgc,'osi_rgc_median': osi_rgc_median,
                  'dsi_sc_mean': dsi_sc,'dsi_sc_median': dsi_sc_median,
                  'osi_sc_mean': osi_sc,'osi_sc_median': osi_sc_median,
                  'r_sc': r_sc} 
    
    return chirp_color_sc,chirp_color_rgc,dsi_osi_dict



















