import numpy as np
import generative


def compute_confusion_matrix(predictions, LTE):
    conf_matrix = np.zeros((2, 2))
    for i in range(len(LTE)):
        conf_matrix[predictions[i], LTE[i]] += 1
    return conf_matrix

def compute_DCF(pi, Cfn, Cfp, llrs, LTE, threshold = None):
    
    if threshold is None:
        threshold = - np.log(pi * Cfn / ((1-pi) * Cfp))
    
    predictions = (llrs > threshold).astype(int)
    conf_matrix = np.zeros((2, 2))
    for i in range(len(LTE)):
        conf_matrix[predictions[i], LTE[i]] += 1
    
    #print("Confusion matrix:\n")
    #print(conf_matrix)
    
    Pfn = conf_matrix[0,1] / (conf_matrix[0,1] + conf_matrix[1, 1])
    Pfp = conf_matrix[1,0] / (conf_matrix[1,0] + conf_matrix[0, 0])
    
    DCF_u = pi * Cfn * Pfn + (1 - pi) * Cfp * Pfp
    #print(f"\nPrior: {pi}, Cfp: {Cfp}, Cfn: {Cfn}, DCF_u : {DCF_u}\n")
    
    return DCF_u, conf_matrix

def compute_DCF_normalized(pi, Cfn, Cfp, llrs, LTE, threshold = None):
    dcf_u, conf_matrix = compute_DCF(pi, Cfn, Cfp, llrs, LTE, threshold)
    dummy_risk = np.min([pi * Cfn, (1 - pi) * Cfp])
    return (dcf_u / dummy_risk), conf_matrix

def compute_min_DCF(pi, Cfn, Cfp, llrs, LTE):
    sorted_llrs = np.sort(llrs)
    all_possible_dcf = []
    for t in sorted_llrs:
        dcf_el, _ = compute_DCF_normalized(pi, Cfn, Cfp, llrs, LTE, t)
        all_possible_dcf.append(dcf_el)
        
    return np.min(all_possible_dcf)

def compute_actual_DCF(pi, llrs, LTE):
    actDCF, conf_matrix = compute_DCF_normalized(pi, 1, 1, llrs, LTE, None)    
    return actDCF, conf_matrix
    

def evaluate(name, config, llrs, LTE):
    pi, Cfn, Cfp = config
    DCF, conf_matrix = compute_DCF_normalized(pi, Cfn, Cfp, llrs, LTE)
    min_DCF = compute_min_DCF(pi, Cfn, Cfp, llrs, LTE)
    act_dcf, _ = compute_actual_DCF(pi, llrs, LTE)
    print(f"Model: {name} with prior: {pi}")
    print("MinDCF: %.4f "% min_DCF)
    print("Actual DCF: %.4f " % act_dcf)
    return DCF, min_DCF

def bayesPlot(S, L, left = -3, right = 3, npts = 21):
    
    effPriorLogOdds = np.linspace(left, right, npts)
    effPriors = 1.0 / (1.0 + np.exp(-effPriorLogOdds))
    actDCF = []
    minDCF = []
    for effPrior in effPriors:
        actDCF.append(compute_actual_DCF(effPrior, S, L)[0])
        minDCF.append(compute_min_DCF(effPrior, 1, 1, S, L))
    return effPriorLogOdds, minDCF, actDCF


def compute_optimal_Bayes_binary_llr(llr, prior, Cfn, Cfp):
    th = -np.log( (prior * Cfn) / ((1 - prior) * Cfp) )
    return np.int32(llr > th)