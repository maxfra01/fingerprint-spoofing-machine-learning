import numpy as np
import scipy.linalg


#### GLOBAL VARIABLES ####
FEATURE_NUMBER = 6


def load(FILENAME):
   d = []
   l = []
   with open(FILENAME, 'r') as f:
      line = f.readline()
      while line != "":
         fields = line.split(",")
         features = fields[0:FEATURE_NUMBER]
         label = fields[6]
         d.append(features)
         l.append(label)
         line = f.readline()
   D = np.array(d, dtype=np.float64).T
   L = np.array(l, dtype=np.int32).ravel()
   return D,L

def vcol(X):
   return X.reshape(X.shape[0], 1)

def vrow(X):
   return X.reshape(1, X.shape[0])

def compute_cov_matrix(D):
   mu = np.mean(D, axis=1, dtype=np.float64).reshape(D.shape[0], 1)
   data_centered = D - mu 
   covariance_matrix = (data_centered @ data_centered.T) / float(D.shape[1])
   return covariance_matrix

def center_data(D):
   mu = np.mean(D, axis=1, dtype=np.float64).reshape(D.shape[0], 1)
   return D - mu

def compute_correlation_matrix(D, L):
   C0 = compute_cov_matrix(D[:, L==0])
   C1 = compute_cov_matrix(D[:, L==1])
   Corr0 = C0 / (vcol(np.diag(C0) ** 0.5) * vrow(np.diag(C0) ** 0.5))
   Corr1 = C1 / (vcol(np.diag(C1) ** 0.5) * vrow(np.diag(C1) ** 0.5))
   
   #print("Correlation matrix for class 0: \n", Corr0)
   #print("Correlation matrix for class 1: \n", Corr1)
   
   return (Corr0, Corr1)

def compute_mu_C(D):
    mu = vcol(D.mean(1))
    C = ((D-mu) @ (D-mu).T) / float(D.shape[1])
    return mu, C

def pca(D, m):
   mu, C = compute_mu_C(D)
   U, s, Vh = np.linalg.svd(C)
   P = U[:, 0:m]
   return P

def compute_mu_C(D): # Same as in pca script
    mu = vcol(D.mean(1))
    C = ((D-mu) @ (D-mu).T) / float(D.shape[1])
    return mu, C

def compute_Sb_Sw(D, L):
    Sb = 0
    Sw = 0
    muGlobal = vcol(D.mean(1))
    for i in np.unique(L):
        DCls = D[:, L == i]
        mu = vcol(DCls.mean(1))
        Sb += (mu - muGlobal) @ (mu - muGlobal).T * DCls.shape[1]
        Sw += (DCls - mu) @ (DCls - mu).T
    return Sb / D.shape[1], Sw / D.shape[1]

def compute_lda_geig(D, L, m):
    
    Sb, Sw = compute_Sb_Sw(D, L)
    s, U = scipy.linalg.eigh(Sb, Sw)
    return U[:, ::-1][:, 0:m]

def compute_lda_JointDiag(D, L, m):

    Sb, Sw = compute_Sb_Sw(D, L)

    U, s, _ = np.linalg.svd(Sw)
    P = np.dot(U * vrow(1.0/(s**0.5)), U.T)

    Sb2 = np.dot(P, np.dot(Sb, P.T))
    U2, s2, _ = np.linalg.svd(Sb2)

    P2 = U2[:, 0:m]
    return np.dot(P2.T, P).T

def apply_lda(U, D):
    return U.T @ D

def split_db_2to1(D, L, seed=0):
   nTrain = int(D.shape[1]*2.0/3.0)
   np.random.seed(seed)
   idx = np.random.permutation(D.shape[1])
   idxTrain = idx[0:nTrain]
   idxTest = idx[nTrain:]
   DTR = D[:, idxTrain] 
   DVAL = D[:, idxTest]
   LTR = L[idxTrain]
   LVAL = L[idxTest] 
   return (DTR, LTR), (DVAL, LVAL)

def predict_label(llr, threshold):
   prediction = np.zeros(llr.shape[0])
   for i in range(llr.shape[0]):
      if llr[i] > threshold:
         prediction[i] = 1
      else:
         prediction[i] = 0
   return prediction

def compute_error_rate(prediction, L):
   return np.sum(prediction != L) / float(L.shape[0])

KFOLD = 5
def extract_train_val_folds_from_ary(X, idx):
    return np.hstack([X[jdx::KFOLD] for jdx in range(KFOLD) if jdx != idx]), X[idx::KFOLD]
