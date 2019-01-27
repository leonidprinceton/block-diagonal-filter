#MIT License
#
#Copyright (c) 2019, Leonid Pogorelyuk
#
#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:
#
#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.

import numpy as np
import time

##
#
# This example compares the Kalman filter (https://en.wikipedia.org/wiki/Kalman_filter#Details)
# to the block-diagonal filter (to appear on arXiv soon). It considers N_SYSTEMS identical
# 2-dimensional linear discrete-time dynamical systems,
# 
# x_i_1(t) = alpha*x_i_1(t-1) +     1*x_i_2(t-1) + v_i_1(t) + G_i_1*v_bar(t)
# x_i_2(t) =                  + alpha*x_i_2(t-1) + v_i_2(t) + G_i_2*v_bar(t)
#
# y_i(t)   = x_i_1(t)         +       x_i_2(t)   + w_i(t)
#
# where:
# - i is the index of the system
# - x_i_1(t), x_i_2(t) is the state of the system i at time t
# - y_i(t) is the measurement of system i at time t
# - v_i_1(t), v_i_2(t) are the first part of the process noise of system i at time t;
#                      they are uncorrelated between different systems and times;
#                      they are normally distributed with zero mean and covariance Q = beta*I (I is a 2-by-2 identity matrix)
# - v_bar(t) are second part of the process noise and are shared by all systems (acting as coupling inputs);
#            they are uncorrelated between different times and uncorrelated with v_i_1(t), v_i_2(t);
#            they are normally distributed with zero mean and covariance Q_bar;
# - w_i(t) are measurement noise of the system i;
#          they are uncorrelated between different systems and times, and uncorrelated with any of the above noise terms;
#          they are normally distributed with zero mean and covariance R
# - alpha, beta, G_i_1, G_i_2 are constants
#
# The goal of the filters is to estimate x_i_1(t), x_i_2(t) (for all i) given the measurements y_i(0),y_i(1),...,y_i(t).
# The Kalman filter is the "optimal" recursive estimator and its computational complexity is O(N_SYSTEMS^3) per time step.
# The block-diagonal filter is sub-optimal. It exploits the above systems structure to compute its estimates in O(N_SYSTEMS).
# For comparison purposes, the code below includes also an "artificially-decoupled" filter. It consists of N_SYSTEMS 
# independent Kalman filters, treating v_bar(t) (the coupling inputs) as if they were different and independent for each system.
#
# For the particular choice of constants in this example:
# - the Kalman filter is the most accurate, and the block-diagonal filter is almost as accurate
# - the "artificially-decoupled" filter is significantly less accurate
# - the "artificially-decoupled" filter is the fastest (with O(N_SYSTEMS) complexity), followed by the block-diagonal filter
# - the Kalman filter is significantly slower
# Therefore, the block-diagonal filter is arguably the most "cost-effective" of the three, computing almost optimal
# estimates in a relatively short time.
#
# The dependency of the block-diagonal filter on the parameters is as follows:
# - as N_SYSTEMS grows, its estimates become closer to those of the Kalman filter
# - as beta decreases, its estimates become more accurate compared to the "artificially-decoupled" filter
# - for alpha above some critical value (around 1.5), the block-diagonal filter becomes very inaccurate
#
##

## number of systems and their dimensions
N_SYSTEMS = 128
STATE_DIM = 2
MEASUREMENT_DIM = 1
COUPLING_DIM = 1

## 
# The dynamics equations above, are formulated in vector form as follows:
# x_i(t)   = F*x_i(t-1)  +  v_i(t)  +  G_i*v_bar(t)
# y_i(t)   = H*x_i(t-1)  +  w_i(t)
# v_i(t)   ~ N(0, Q)
# v_bar(t) ~ N(0, Q_bar)
# w_i(t)   ~ N(0, R)
# where F,H,Q,Q_bar and R are defined below, and G_i = [[1],[1]].
##
alpha = 1.0
beta = 1.0/256
F = np.array([[alpha, 1], [0, alpha]])
H = np.ones((MEASUREMENT_DIM, STATE_DIM))
Q = beta*np.eye(STATE_DIM)
R = np.eye(MEASUREMENT_DIM)
assert(F.shape == Q.shape)
Q_bar = np.eye(COUPLING_DIM)
G = np.ones((N_SYSTEMS*STATE_DIM, COUPLING_DIM)) #G consists of all G_i stacked together

## initial state distributions  (identical for all systems), x_i ~ N(0, P0) with P0 = sigma_0^2*I
sigma_0 = 32
x0_mean = np.zeros(STATE_DIM)
P0 = sigma_0**2*np.eye(STATE_DIM)

## initial states; all x_i are stacked into x
np.random.seed(0)
x = np.random.multivariate_normal(x0_mean, P0, N_SYSTEMS).flatten()

##
# The dynamics of the "full" system (all the identical sub-systems combined), are
# x(t) = F_full*x(t-1)  +  v(t)  +  G*v_bat(y)
# y(t) = H_full*x(t)    +  w(t)
# v    ~ N(0,Q_full)
# w    ~ N(0,R_full)
# with the compound matrices (which are also used by the Kalman filter) defined as follows:
##
F_full = np.kron(np.eye(N_SYSTEMS), F) #F_full.shape is (N_SYSTEMS*STATE_DIM, N_SYSTEMS*STATE_DIM)
H_full = np.kron(np.eye(N_SYSTEMS), H)
R_full = np.kron(np.eye(N_SYSTEMS), R)
Q_full = np.kron(np.eye(N_SYSTEMS), Q) + G.dot(Q_bar).dot(G.T)
P_full = np.kron(np.eye(N_SYSTEMS), P0) #initial Kalman filter covariance
x_full = np.repeat(x0_mean, N_SYSTEMS)  #initial estimate

## the block-diagonal filter stores just the non-zero block-entries of the above matrices
F_block = np.repeat([F], N_SYSTEMS, axis=0) #F_block.shape is (N_SYSTEMS, STATE_DIM, STATE_DIM)
H_block = np.repeat([H], N_SYSTEMS, axis=0)
R_block = np.repeat([R], N_SYSTEMS, axis=0)
Q_block = np.repeat([Q], N_SYSTEMS, axis=0)
G_block = G.reshape((N_SYSTEMS, STATE_DIM, COUPLING_DIM))
P_block = np.repeat([P0], N_SYSTEMS, axis=0)
x_block = np.repeat(x0_mean, N_SYSTEMS, axis=0).reshape((N_SYSTEMS, STATE_DIM, 1))

## the "artificially-decoupled" uses the same block-entries as the block-diagonal filter, except Q
G_block_trans = G_block.transpose(0,2,1)
Q_decoupled = Q_block + np.matmul(G_block.dot(Q_bar), G_block_trans) #assumes G_i*v_bar(t) are uncorrelated between different i
P_decoupled = np.repeat([P0], N_SYSTEMS, axis=0)
x_decoupled = np.repeat(x0_mean, N_SYSTEMS, axis=0).reshape((N_SYSTEMS, STATE_DIM, 1))

## the example keeps track of the quantities below for N_ITER time steps
N_ITER = 64 #the number of time steps is relatively small since for alpha>1 numerical errors become dominant
errors_full = np.zeros(N_ITER)              #estimate errors of the Klaman filter
errors_block = np.zeros(N_ITER)             #estimate errors of the block diagonal filter
errors_decoupled = np.zeros(N_ITER)         #estimate errors of the "artificially-decoupled" filter
estimates_diff_block = np.zeros(N_ITER)     #difference between the estimates of the Klaman and block diagonal filters
estimates_diff_decoupled = np.zeros(N_ITER) #difference between the estimates of the Klaman and "artificially-decoupled" filters
time_full      = 0 #total time spent advancing the Kalman filter (in seconds)
time_block     = 0 #total time spent advancing the block-diagonal filter
time_decoupled = 0 #total time spent advancing the "artificially-decoupled" filter

for t in range(N_ITER):
	## advancing the state of the full system
	v_bar = np.random.multivariate_normal(np.zeros(COUPLING_DIM), Q_bar) #correlated process noise (coupling input)
	v = np.random.multivariate_normal(np.zeros(STATE_DIM), Q, N_SYSTEMS).flatten() #uncorrelated process noise
	x = F_full.dot(x) + v + G.dot(v_bar)

	## performing measurements
	w = np.random.multivariate_normal(np.zeros(MEASUREMENT_DIM), R, N_SYSTEMS).flatten() #uncorrelated measurement noise
	y = H_full.dot(x) + w

	## advancing the (full) Kalman filter
	start_time = time.time()
	x_full = F_full.dot(x_full)
	y_full = H_full.dot(x_full)
	P_full = F_full.dot(P_full).dot(F_full.T) + Q_full
	S_full = H_full.dot(P_full).dot(H_full.T) + R_full
	S_full_inv = np.linalg.inv(S_full)
	K_full = P_full.dot(H_full.T).dot(S_full_inv)
	x_full = x_full + K_full.dot(y.reshape(y_full.shape) - y_full)
	P_full = P_full - K_full.dot(H_full).dot(P_full)
	time_full += time.time() - start_time

	## advancing the "artificially-decoupled" Filter (same equations as above propagated along of axes 1 and 2)
	start_time = time.time()
	F_block_trans = F_block.transpose(0,2,1) #although F and H are constant here, they can be time dependent in general
	H_block_trans = H_block.transpose(0,2,1)
	x_decoupled = np.matmul(F_block, x_decoupled)
	y_decoupled = np.matmul(H_block, x_decoupled)
	P_decoupled = np.matmul(np.matmul(F_block, P_decoupled), F_block_trans) + Q_decoupled
	S_decoupled = np.matmul(np.matmul(H_block, P_decoupled), H_block_trans) + R_block
	S_decoupled_inv = np.linalg.inv(S_decoupled)
	K_decoupled = np.matmul(np.matmul(P_decoupled, H_block_trans), S_decoupled_inv)
	x_decoupled = x_decoupled + np.matmul(K_decoupled, y.reshape(y_decoupled.shape) - y_decoupled)
	P_decoupled = P_decoupled - np.matmul(np.matmul(K_decoupled, H_block), P_decoupled)
	time_decoupled += time.time() - start_time

	## advancing the block diagonal Filter (a paper on arXiv will be uploaded soon)
	start_time = time.time()
	HG_block = np.matmul(H_block, G_block)
	HG_block_trans = HG_block.transpose(0,2,1)

	x_block = np.matmul(F_block, x_block)
	y_block = np.matmul(H_block, x_block)

	L_block = np.matmul(np.matmul(F_block, P_block), F_block_trans) + Q_block
	M_block = np.matmul(np.matmul(H_block, L_block), H_block_trans) + R_block
	M_block_inv = np.linalg.inv(M_block)
	N_block = np.matmul(np.matmul(HG_block_trans, M_block_inv), HG_block)
	N = np.sum(N_block, axis=0)
	C1 = np.linalg.inv(np.linalg.inv(Q_bar) + N)
	C2 = Q_bar.dot(N.dot(C1).dot(N) - N).dot(Q_bar) + Q_bar
	C3 = Q_bar.dot(N).dot(C1) - Q_bar
	LH_block = np.matmul(L_block, H_block_trans)
	HL_block = np.matmul(H_block, L_block)
	A_block = L_block - np.matmul(np.matmul(LH_block, M_block_inv), HL_block)
	B_block = np.matmul(np.matmul(LH_block, M_block_inv), HG_block)
	B_block_trans = B_block.transpose(0,2,1)

	_x_block_update_term_1 = np.matmul(H_block_trans, np.matmul(M_block_inv, y.reshape(y_block.shape) - y_block))
	_x_block_update_term_2 = np.sum(np.matmul(G_block_trans, _x_block_update_term_1), axis=0)
	x_block = x_block + np.matmul(L_block, _x_block_update_term_1) - (B_block.dot(C1) + G_block.dot(C3)).dot(_x_block_update_term_2)
	P_block = A_block + np.matmul(B_block.dot(C1), B_block_trans) + np.matmul(G_block.dot(C2), G_block_trans) + np.matmul(B_block.dot(C3), G_block_trans) + np.matmul(G_block.dot(C3), B_block_trans)
	time_block += time.time() - start_time
	
	## keeping track of sums of squares of errors and differences between estimates
	errors_full[t]              = np.sum((x.reshape(x_full.shape)      - x_full)**2)
	errors_block[t]             = np.sum((x.reshape(x_block.shape)     - x_block)**2)
	errors_decoupled[t]         = np.sum((x.reshape(x_decoupled.shape) - x_decoupled)**2)
	estimates_diff_block[t]     = np.sum((x_full.reshape(x_block.shape)     - x_block)**2)
	estimates_diff_decoupled[t] = np.sum((x_full.reshape(x_decoupled.shape) - x_decoupled)**2)

## printing results
SETTLING_TIME = N_ITER//2 #the empirical time it takes the filters to "reach" steady-state (become "asymptotic")
avg_err_full      = np.mean(errors_full[SETTLING_TIME:])
avg_err_block     = np.mean(errors_block[SETTLING_TIME:])
avg_err_decoupled = np.mean(errors_decoupled[SETTLING_TIME:])
avg_est_diff_block = np.mean(estimates_diff_block[SETTLING_TIME:])
avg_est_diff_decoupled = np.mean(estimates_diff_decoupled[SETTLING_TIME:])
print("Average squared error of the asymptotic Kalman filter:                % 6.2f, (%.3f per sub-system)"%(avg_err_full, avg_err_full/N_SYSTEMS))
print("Average squared error of the asymptotic block-diagonal filter:        % 6.2f, (%.3f per sub-system)"%(avg_err_block, avg_err_block/N_SYSTEMS))
print("Average squared error of the asymptotic artificially-decoupled filter:% 6.2f, (%.3f per sub-system)"%(avg_err_decoupled, avg_err_decoupled/N_SYSTEMS))
print("")
print("Average squared difference between the estimates of the Kalman and block-diagonal filters:        % 6.2f, (%.3f per sub-system)"%(avg_est_diff_block, avg_est_diff_block/N_SYSTEMS))
print("Average squared difference between the estimates of the Kalman and artificially-decoupled filters:% 6.2f, (%.3f per sub-system)"%(avg_est_diff_decoupled, avg_est_diff_decoupled/N_SYSTEMS))
print("")
P_Kalman = P_full[:STATE_DIM,:STATE_DIM] #all systems are identical, so only the first one is considered
ref_diff_cov_block = np.linalg.norm(P_block[0] - P_Kalman)/np.linalg.norm(P_Kalman)
ref_diff_cov_decoupled = np.linalg.norm(P_decoupled[0] - P_Kalman)/np.linalg.norm(P_Kalman)
print("Relative difference between the covariance of the Kalman filter and its approximation by the block-diagonal filter:        % 6.3f per sub-system"%(ref_diff_cov_block))
print("Relative difference between the covariance of the Kalman filter and its approximation by the artificially-decoupled filter:% 6.3f per sub-system"%(ref_diff_cov_decoupled))
print("")
print("Time per iteration of the Kalman filter:                %6.3fms"%(time_full/N_ITER*1000))
print("Time per iteration of the block-diagonal filter:        %6.3fms"%(time_block/N_ITER*1000))
print("Time per iteration of the artificially-decoupled filter:%6.3fms"%(time_decoupled/N_ITER*1000))
