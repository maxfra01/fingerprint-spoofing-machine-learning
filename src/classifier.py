import numpy as np
import utilities
import plots
import generative

pca_possible_values = [6,5,4,3,2,1]

def pca_with_mvg(DTR,LTR,DVAL,LVAL):
    best_error_rate = 1
    best_pca_value = 6
    
    
    for pca_value in pca_possible_values:
        U_pca = utilities.pca(DTR, pca_value)
        DTR_pca = np.dot(U_pca.T, DTR)
        DVAL_pca = np.dot(U_pca.T, DVAL)
    
        #W= utilities.lda(DTR_pca,LTR, 1)
        #DTR_lda =  np.dot(W.T, DTR_pca)
        #DVAL_lda =  np.dot(W.T, DVAL_pca)
        
        print("***********************PCA value: ", pca_value, "************************")        

        generative.mvg_classifier(DTR_pca, LTR, DVAL_pca, LVAL)
        generative.tied_covariance_gaussian_classifier(DTR_pca, LTR, DVAL_pca, LVAL)
        generative.naive_bayes_classifier(DTR_pca, LTR, DVAL_pca, LVAL)                
    
    return (best_pca_value, best_error_rate)

def lda_with_pca(DTR, LTR, DVAL, LVAL):
    
    
    for pca_value in pca_possible_values:
        U_pca = utilities.pca(DTR, pca_value)
        DTR_pca = np.dot(U_pca.T, DTR)
        DVAL_pca = np.dot(U_pca.T, DVAL)
        W = utilities.compute_lda_geig(DTR_pca, LTR, 1)
        DTR_lda = np.dot(W.T, DTR_pca)
        DVAL_lda = np.dot(W.T, DVAL_pca)
        best_error_rate = 1
        best_threshold= 0

        print("***********************PCA value: ", pca_value, "************************")
        threshold = (DTR_lda[0, LTR==0].mean() + DTR_lda[0, LTR==1].mean()) / 2.0
        if DTR_lda[0, LTR== 0].mean() > DTR_lda[0, LTR==1].mean():
            DVAL_lda = -DVAL_lda
        for th in np.linspace(threshold-1, threshold+1, 1000):
            PVAL = np.zeros(shape=LVAL.shape, dtype=np.int32)
            PVAL[DVAL_lda[0] >= threshold] = 1
            PVAL[DVAL_lda[0] < threshold] = 0  
            error_rate = np.sum(PVAL != LVAL) / len(LVAL)
            if error_rate < best_error_rate:
                best_error_rate = error_rate
                best_pca_value = pca_value
                best_threshold = th
        print("Best threshold: ", best_threshold)
        print("Error rate: ", best_error_rate)
            