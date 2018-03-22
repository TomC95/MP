# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 22:38:59 2017

Generates SV time series with phi = 0 and attempts to calculate parameters of
time series.

Known bugs:
    None

@author: Thomas
"""
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def time_series(T, mu, sigma_eta):
    """
    Generates times series of length T when function is called.
    
    Notes: Seems to work correctly.
    """
    y = []
    for t in range(T):
        epsilon = np.random.normal(0, 1)
        eta = np.random.normal(0, sigma_eta)
        h_t = mu + eta
        y.append(np.exp(h_t/2)*epsilon)
    return y

"""
Generating Time Series:
"""

T = int(input('Enter number of steps for time series: '))
mu = -1
sigma_eta = np.sqrt(0.05)
y = np.array(time_series(T, mu, sigma_eta))


"""
Functions used in Monte Carlo simulation:
"""


def p_overall(T, y_t, sigma_eta_sq, mu, h_t):
    """
    Calculations for Equations 10 and 11 from Takaishi paper with phi = 0. Used
    for ratio of probabilities for global accept-reject step.
    """
    sigma_t = np.exp(h_t/2)
    list1 = -(y_t**2)/(2*sigma_t**2) + np.log((2*np.pi*sigma_t**2)**(-1/2))
    list2 = -((h_t-mu)**2)/(2*sigma_eta_sq) + np.log((2*np.pi*sigma_eta_sq)**(-1/2))
    likelihood = (np.sum(list1 + list2))
    #print("Likelihood = {}".format(likelihood))
    prior = 1.0/(sigma_eta_sq)
    return likelihood, prior


def inv_gamma(h_t, mu_1, T):
    """
    Calculates Equations 12 and 13 from Takaishi paper with phi = 0.
    """
    A = 0.5*(np.sum((h_t - mu_1)**2))
    inv = stats.invgamma.rvs(T/2, scale = A)
    #print(A, inv)
    return inv


def metropolis_hastings_h_t(x, mu, sigma_eta, y_t):
    """
    Performs Metropolis-Hastings algorithm with initial values x, mean mu,
    standard deviation sigma_eta and data y_t. Outputs list of accepted values
    xt.
    """
    xt = x
    x_cand = np.random.normal(mu, sigma_eta, T)
    ht_cur = (-0.5)*(x_cand + (y_t**2)/np.exp(x_cand))
    ht_prev = (-0.5)*(x + (y_t**2)/np.exp(x))
    alpha = np.exp(ht_cur - ht_prev)
    rand_u_list = np.random.uniform(0, 1, T)
    ind = np.where((rand_u_list<=alpha))
    print('Indexes Length: {}'.format(len(ind[0])))
    xt[ind[0]] = x_cand[ind[0]]
    return xt


"""
Set-up for process:
"""

sigma_eta_sq_init = 0.5 #sigma_eta*np.random.uniform(0.8, 1.2)
mu_init = 1 #mu*np.random.uniform(0.8, 1.2)
h_t_init  = np.array([np.random.normal() for i in range(T)]) #= np.ones(T)
print("Inital values: sigma_eta = {}, mu = {}".format(sigma_eta_sq_init, mu_init))
sigma_eta_sq_n = sigma_eta_sq_init
mu_n = mu_init
h_t_n = np.array(list(h_t_init))
N = 4000
xt_list = []
ar_list = []
test_list = []
cand_values_list = []
test_list0 = []

"""
Monte Carlo Simulation process:
"""

start_time = time.time()
for n in range(N):
    sigma_eta_sq_cand = inv_gamma(h_t_n, mu_n, T)
    mu_cand = np.random.normal((1/T)*np.sum(h_t_n), (np.sqrt(sigma_eta_sq_n)*np.sqrt(1/T)))
    h_t_cand = metropolis_hastings_h_t(h_t_n, mu_n, np.sqrt(sigma_eta_sq_n), y)
    prob_cand = p_overall(T, y, sigma_eta_sq_cand, mu_cand, h_t_cand)
    print(prob_cand[0])
    prob_old = p_overall(T, y, sigma_eta_sq_n, mu_n, h_t_n)
    print(prob_old[0])
    cand_values = [sigma_eta_sq_cand, mu_cand, h_t_cand]
    like_comp = prob_cand[0] - prob_old[0]
    prior_comp = prob_cand[1]/prob_old[1]
    test_value = np.exp((like_comp))*(prior_comp)
    alpha = np.min([1, test_value])
    rand_u = np.random.uniform(0, 1)
    if rand_u < alpha:
        print("Accept = YES")
        sigma_eta_sq_n = sigma_eta_sq_cand
        mu_n = mu_cand
        h_t_n = np.array(list(h_t_cand))
        ar_list.append(1)
    else:
        h_t_n = h_t_n
    xt_list.append([sigma_eta_sq_n, mu_n, h_t_n])
    test_list.append(test_value)
    cand_values_list.append(cand_values)
    print(n)
print(xt_list[N-1])
print("Acceptance ratio for global step is {} .".format(len(ar_list)/N))
print("--- {} seconds ---".format(time.time() - start_time))

"""
End of Simulation: Presentation of results below.
"""

sigma_eta_sq_n_final = [xt_list[i][0] for i in range(N)]
sigma_eta_n_final = [np.sqrt(i) for i in sigma_eta_sq_n_final]
mu_n_final = [xt_list[i][1] for i in range(N)]
h_t_final = [xt_list[i][2] for i in range(N)]
h_10_final = [h_t_final[i][9] for i in range(N)]

sigma_eta_sq_n_cutoff = sigma_eta_sq_n_final[999:]
sigma_eta_n_cutoff = sigma_eta_n_final[999:]
mu_n_cutoff = mu_n_final[999:]
h_t_cutoff = h_t_final[999:]
h_10_cutoff = h_10_final[999:]

mu_n_mean = np.mean(mu_n_cutoff)
mu_n_std = np.std(mu_n_cutoff)
mu_n_diff = abs(mu - mu_n_mean)
print("The mean of mu is: {}".format(mu_n_mean))
print("The standard deviation of mu is: {}".format(mu_n_std))
print("The error in mu is: {}".format(mu_n_diff))

sigma_eta_sq_n_mean = np.mean(sigma_eta_sq_n_cutoff)
sigma_eta_sq_n_std = np.std(sigma_eta_sq_n_cutoff)
sigma_eta_sq_n_diff = abs(sigma_eta**2 - sigma_eta_sq_n_mean)
print("The mean of sigma_eta is: {}".format(sigma_eta_sq_n_mean))
print("The standard deviation of sigma_eta is: {}".format(sigma_eta_sq_n_std))
print("The error in sigma_eta squared is: {}".format(sigma_eta_sq_n_diff))

x = [n for n in range(N)]
plt.figure()
plt.plot(x, sigma_eta_sq_n_final, label='Result')
plt.plot(x, [sigma_eta_sq_n_mean for i in range(N)], label='Mean')
plt.plot(x, [sigma_eta**2 for i in range(N)], label='Actual')
plt.title("Sigma_eta_sq time series")
plt.legend()
plt.figure()
plt.plot(x, mu_n_final, label='Result')
plt.plot(x, [mu_n_mean for i in range(N)], label='Mean')
plt.plot(x, [mu for i in range(N)], label='Actual')
plt.title("Mu time series")
plt.legend()
plt.figure()
plt.plot(x, h_10_final)
plt.title("h_10 time series")
plt.show()
