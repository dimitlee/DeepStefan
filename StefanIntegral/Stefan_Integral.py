# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 20:27:53 2020

@author: sifan
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatter
from scipy.interpolate import griddata
import seaborn as sns
from .Stefan_models_tf import Sampler, DataSampler, StefanIntegral
import pandas as pd
import os


def run_stefan_integral():

    # Exact Solution
    def u(z):
        # x = (x, t)
        x = z[:, 0: 1]
        t = z[:, 1: 2]
        u = - 0.5 * x**2  + 2 * x - t - 0.5
        return u

    # Exact free boundary
    def s(x):
        t = x[:, 1: 2]
        s = 2 - np.sqrt(3 - 2 * t)
        return s

    def h(x):
        # du / dx (s(t), t) = h(t)   Stefan Neumann Condition
        t = x[:, 1: 2]
        return np.sqrt(3 - 2 * t)

    def u_0(x):
        #  Initial Condition
        x = x[:, 0: 1]
        return - 0.5 * x**2 + 2 * x - 0.5
    
    def e(x):
        # integral[0 to s(t)] of u(x, t)dx = e(x)
        # s = lambda t: 2 - np.sqrt(3 - 2 * t)
        t = x[:, 1: 2]
        
        return 5/3 - 2*t + ((2/3)*t - 1) * np.sqrt(3 - 2*t)
        # return - s(t)**3 / 6 + s(t)**2 - (t + 0.5) * s(t)
        

    # Domain boundaries
    ic_coords = np.array([[0.0, 0.0],
                          [1.0, 0.0]])
    Nc_coords = np.array([[0.0, 0.0],
                          [0.0, 1.0]])
    dom_coords = np.array([[0.0, 0.0],
                           [1.0, 1.0]])

    # Create boundary conditions samplers
    ics_sampler = Sampler(2, ic_coords, u_0, name='Initial Condition')
    Ncs_sampler = Sampler(2, Nc_coords, h, name='Stefan Neumann Boundary Condition')  # Moving boundary, only use t_u_tf because x = s(t) is a function of t
    integral_sampler = Sampler(2, Nc_coords, e, name="Integral-type Boundary Condition")

    # Create residual sampler
    res_sampler = Sampler(2, dom_coords, u, name='Forcing')

    # Define model
    layers_u = [2, 100, 100, 100, 1]
    layers_s = [1, 100, 100, 100, 1]  # or we can map s to (t, s(t))
    model = StefanIntegral(layers_u, layers_s, ics_sampler, Ncs_sampler, integral_sampler, res_sampler)

    model.train(nIter=40000, batch_size=128)
    
    ### Save Model ###
    ####################
    # save path
    relative_path = os.path.join('results', 'StefanIntegral')
    current_directory = os.getcwd()
    save_results_to = os.path.join(current_directory, relative_path)

    if not os.path.exists(save_results_to):
       os.makedirs(save_results_to)

    model.save_weights(os.path.join(save_results_to, 'model_weights.ckpt'))
    print(f"Model weights saved to: {os.path.join(save_results_to, 'model_weights.ckpt')}")


    # Test data
    nn = 200
    x = np.linspace(dom_coords[0, 0], dom_coords[1, 0], nn)[:, None]
    t = np.linspace(dom_coords[0, 1], dom_coords[1, 1], nn)[:, None]
    X, T = np.meshgrid(x, t)
    X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
    
    # Exact solutions
    u_star = u(X_star)
    s_star = s(X_star)
    
    # Predictions
    u_pred = model.predict_u(X_star)
    s_pred = model.predict_s(X_star)
    
    # Errors
    error_u = np.linalg.norm(u_star - u_pred, 2) / np.linalg.norm(u_star, 2)
    error_s = np.linalg.norm(s_star - s_pred, 2) / np.linalg.norm(s_star, 2)

    print('Relative L2 error_u: {:.2e}'.format(error_u))
    print('Relative L2 error_s: {:.2e}'.format(error_s))
      

    U_star = griddata(X_star, u_star.flatten(), (X, T), method='cubic')
    U_pred = griddata(X_star, u_pred.flatten(), (X, T), method='cubic')
    
    for i in range(nn):
        for j in range(nn):
            X_ij = np.array([X[i,j], T[i,j]]).reshape(1,2)
            u_ij = u(X_ij)
            s_ij = s(X_ij)
            if X[i,j] > s_ij:
                U_star[i,j] = np.nan
                U_pred[i,j] = np.nan
                                
    t = np.linspace(0,1, 100)[:, None]
    x = np.zeros_like(t)
    x_star = np.concatenate((x,t), axis=1)
    
    s_star = s(x_star)
    s_pred = model.predict_s(x_star)
    error_s = np.abs(s_star - s_pred)
                
    fig_1 = plt.figure(1, figsize=(18, 5))
    plt.subplot(1, 3, 1)
    plt.plot(s_star, t)
    plt.pcolor(X, T, U_star, cmap='jet')
    plt.colorbar()
    plt.xlabel(r'$x$')
    plt.ylabel(r'$t$')
    plt.title('Exact $u(x,t)$')
    
    plt.subplot(1, 3, 2)
    plt.pcolor(X, T, U_pred, cmap='jet')
    plt.plot(s_pred, t)
    plt.colorbar()
    plt.xlabel(r'$x$')
    plt.ylabel(r'$t$')
    plt.title('Predicted $u(x,t)$')

    plt.subplot(1, 3, 3)
    plt.pcolor(X, T, np.abs(U_star - U_pred), cmap='jet')
    plt.colorbar(format='%.0e')
    plt.xlabel(r'$x$')
    plt.ylabel(r'$t$')
    plt.title('Absolute Error')
    
    plt.tight_layout()
    plt.show()
    
    fig_2 = plt.figure(2, figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(t, s_star, label='Exact')
    plt.plot(t, s_pred, '--', label='Predicted')
    plt.xlabel(r'$t$')
    plt.ylabel(r'$s(t)$')
    plt.title('Moving Boundary')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(t, error_s)
    plt.xlabel(r'$t$')
    plt.ylabel(r'Point-wise Error')
    plt.title('Absolute Error')
    plt.yscale('log')
    plt.tight_layout()
    plt.show()

    

    

