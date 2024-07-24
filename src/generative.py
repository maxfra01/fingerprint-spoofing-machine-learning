import numpy as np
import utilities
import scipy.special as sps

PRIOR = 0.1

"""
This function computes the log probability density function (pdf) of a multivariate
Gaussian distribution for a given set of data points.
Arguments X,mu,C must be nparray
"""
def logpdf_GAU_ND(X,mu,C):
   Y = np.empty([1, X.shape[1]])
   for i in range(X.shape[1]):
      x = X[:, i:i+1]
      _sign , det = np.linalg.slogdet(C)
      inv = np.linalg.inv(C)
      diff = x-mu.reshape([X.shape[0],1])
      dot_products = np.matmul(diff.T, np.matmul(inv, diff))
      Y[:,i] = -0.5 * (X.shape[0] * np.log(2 * np.pi) + det +  dot_products)      
   return Y.ravel()


"""
This function computes the logarithm of likelihood function
Arguments X,m_ML,C_ML must be nparray 
"""
def loglikelihood(X, m_ML, C_ML):
   Y = logpdf_GAU_ND(X,m_ML,C_ML)
   return np.sum(Y)


def mvg_classifier(DTR, LTR, DTE, LTE):
   mu0 = DTR[:, LTR==0].mean(axis=1)
   mu1 = DTR[:, LTR==1].mean(axis=1)
    
   C0 = utilities.compute_cov_matrix(DTR[:, LTR==0])
   C1 = utilities.compute_cov_matrix(DTR[:, LTR==1])
    
   logS = np.zeros([2, DTE.shape[1]])
   for i in range(DTE.shape[1]):
      logS[0, i] = loglikelihood(DTE[:, i:i+1], mu0, C0) + np.log(PRIOR)
      logS[1, i] = loglikelihood(DTE[:, i:i+1], mu1, C1) + np.log(PRIOR)
    
   llr = logS[1, :] - logS[0, :]
    
   threshold = 0 
   
   prediction = utilities.predict_label(llr, threshold)
   error_rate = utilities.compute_error_rate(prediction, LTE)
    
   print("----------MVG Classifier----------\n", f"Error rate: {error_rate}\n")
   
   return llr

def tied_covariance_gaussian_classifier(DTR, LTR, DTE, LTE):
   mu0 = DTR[:, LTR==0].mean(axis=1)
   mu1 = DTR[:, LTR==1].mean(axis=1)
    
   C0 = utilities.compute_cov_matrix(DTR[:, LTR==0])
   C1 = utilities.compute_cov_matrix(DTR[:, LTR==1])
    
   C = 1/DTR.shape[1] *( C0*DTR[:, LTR==0].shape[1] + C1*DTR[:, LTR==1].shape[1] )

    
   logS = np.zeros([2, DTE.shape[1]])
   for i in range(DTE.shape[1]):
      logS[0, i] = loglikelihood(DTE[:, i:i+1], mu0, C) + np.log(PRIOR)
      logS[1, i] = loglikelihood(DTE[:, i:i+1], mu1, C) + np.log(PRIOR)
        
   llr = logS[1, :] - logS[0, :]
   threshold = 0
   prediction = utilities.predict_label(llr, threshold)
   error_rate = utilities.compute_error_rate(prediction, LTE)
   
   print("----------Tied Covariance Gaussian Classifier----------\n", f"Error rate: {error_rate}\n")
   
   return llr
   
   
def naive_bayes_classifier(DTR, LTR, DTE, LTE):
   mu0 = DTR[:, LTR==0].mean(axis=1)
   mu1 = DTR[:, LTR==1].mean(axis=1)
    
   C0 = utilities.compute_cov_matrix(DTR[:, LTR==0])
   C0 = C0 * np.eye(C0.shape[0])
   C1 = utilities.compute_cov_matrix(DTR[:, LTR==1])
   C1 = C1 * np.eye(C1.shape[0])
    
   logS = np.zeros([3, DTE.shape[1]])
   for i in range(DTE.shape[1]):
      logS[0, i] = loglikelihood(DTE[:, i:i+1], mu0, C0) + np.log(PRIOR)
      logS[1, i] = loglikelihood(DTE[:, i:i+1], mu1, C1) + np.log(PRIOR)
       
   threshold = 0 
   llr = logS[1, :] - logS[0, :]
   predictions = utilities.predict_label(llr, threshold)
   error_rate = utilities.compute_error_rate(predictions, LTE)
   
   print("----------Naive Bayes Classifier----------\n", f"Error rate: {error_rate}\n")
   
   return llr
   