import numpy as np
import spams
import pandas as pd
from scipy.spatial import distance
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt
mpl.style.use('ggplot')
import seaborn as sns

def factorize(X,d):
	u,s,vt = np.linalg.svd(X)
	return u[:,:d].dot(np.diag(s[:d])),vt[:d]

def recover(P,Utilde,lda=0.25,nonneg=False,use_cholesky=False):
	Unorm = np.linalg.norm(Utilde)**2/Utilde.shape[1]
	U = spams.lasso(np.asfortranarray(Utilde),D=np.asfortranarray(P),lambda1=lda*Unorm,mode=1,numThreads=THREADS,cholesky=use_cholesky,pos=nonneg)
	U = np.asarray(U.todense())
	return U

def dot_available(A,X):
	available = np.invert(np.isnan(X))
	Y = np.zeros((A.shape[0],X.shape[1]))
	for j in range(X.shape[1]):
		for i in range(X.shape[0]):
			if available[i,j]:
				Y[:,j] += A[:,i]*X[i,j]
	return Y

def calc_available_rmse(X,X_hat):
	available = np.invert(np.isnan(X.values))
	return (np.linalg.norm((X-X_hat).values[available])/available.sum())**.5

def calc_available_r2(X,X_hat):
	available = np.invert(np.isnan(X.values))
	return (1-distance.correlation(X.values[available].flatten(),X_hat[available].flatten()))**2

def fill_nan_with_avg(X):
	X_fill = np.copy(X)
	not_available = np.isnan(X)
	available = np.invert(not_available)
	for j in range(X.shape[1]):
		col_avg = np.average(X[:,j][available[:,j]])
		X_fill[not_available[:,j],j] = col_avg
	return X_fill


ferret = pd.read_csv('Fonville2014_TableS1.csv',index_col=0,na_values=['*']).replace('<10',5).astype(float)
pre_vaccine = pd.read_csv('Fonville2014_TableS14_PreVaccination.csv',index_col=0,na_values=['*']).replace('<10',5).astype(float)
post_vaccine = pd.read_csv('Fonville2014_TableS14_PostVaccination.csv',index_col=0,na_values=['*']).replace('<10',5).astype(float)


X = np.log10(pre_vaccine)
X_fill = nuclear_norm_solve(X,np.invert(np.isnan(X)))
A = np.load('Design.n-96_m-24_q-3.npy').astype(np.float)
A = A[:,:X.shape[0]]
# Y = dot_available(A,X)
Y = A.dot(X_fill)
for d in range(10,20):
	Utilde,W = factorize(Y,d)
	U = recover(A,Utilde,lda=1e-5)
	X_hat = U.dot(W)
	c = calc_available_r2(X,X_hat)
	print(d,c)



