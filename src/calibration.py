import numpy as np
import logreg
import utilities
import dcf

KFOLD = 5
APPLICAZION_PRIOR = 0.1

def extract_train_val_folds_from_ary(X, idx):
    return np.hstack([X[jdx::KFOLD] for jdx in range(KFOLD) if jdx != idx]), X[idx::KFOLD]

def calibration_score(scores, labels, pT):
    
    calibrated_scores = []
    calibrated_labels = []
    
    for Idx in range(KFOLD):
        SCAL, SVAL = extract_train_val_folds_from_ary(scores, Idx)
        LCAL, LVAL = extract_train_val_folds_from_ary(labels, Idx)
        w,b = logreg.trainWeightedLogRegBinary(utilities.vrow(SCAL), LCAL, 0 , pT)
        calibrated_SVAL = (w.T @ utilities.vrow(SVAL) + b - np.log(pT / (1-pT))).ravel()
        calibrated_scores.append(calibrated_SVAL)
        calibrated_labels.append(LVAL)
        
    calibrated_scores = np.hstack(calibrated_scores)
    calibrated_labels = np.hstack(calibrated_labels)
    
    print("\t\tminDCF(p=0.1), cal.   :  %.4f" % dcf.compute_min_DCF(APPLICAZION_PRIOR, 1,1, calibrated_scores, calibrated_labels))
    print("\t\tActualDCF(p=0.1), cal.:  %.4f" % dcf.compute_actual_DCF(APPLICAZION_PRIOR, calibrated_scores, calibrated_labels)[0])
    return calibrated_scores, calibrated_labels    
        