import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import dcf

def vcol(x):
    return x.reshape((x.size, 1))

def vrow(x):
    return x.reshape((1, x.size))

def logreg_obj_wrap(DTR, LTR, l):
    ZTR = 2* LTR - 1
    def logreg_obj(v):
        w,b = v[0:-1], v[-1]
        S = (np.dot(vcol(w).T, DTR)+ b).ravel()
        J = (l / 2) * (np.linalg.norm(w)**2) + (1 / DTR.shape[1]) * np.sum(  np.logaddexp(0, - ZTR * S) )
        return J
    return logreg_obj
    
# def compute_gradient(DTR,LTR, l):
#     def gradient(v):
#         w,b = v[0:-1], v[-1]
#         ZTR = 2* LTR -1
#         S = (vcol(w).T @ DTR + b).ravel()
#         G = -ZTR / (1 + np.exp(ZTR * S))
#         g1 = (l * w).reshape([w.shape[0],1]) + (1 / DTR.shape[1]) * vrow(G) * DTR
#         g2 = np.sum(G) / DTR.shape[1]
#         return (g1,g2)
#     return gradient

# def compute_gradient_weighted(DTR,LTR, l, nt, nf,   target_prior_pi):
#     def gradient_weighted(v):
#         w,b = v[0:-1], v[-1]
#         ZTR = 2* LTR -1
#         EPS =np.where(ZTR == 1, target_prior_pi / nt, (1 - target_prior_pi) / nf)
#         S = (vcol(w).T @ DTR + b).ravel()
#         G = -ZTR / (1 + np.exp(ZTR * S))
#         g1 = (l * w).reshape([w.shape[0],1]) + (1 / DTR.shape[1]) * vrow(EPS * G) * DTR
#         g2 = np.sum(EPS * G) / DTR.shape[1]
#         return (g1,g2)
#     return gradient_weighted
        

def optimize(f, x0, gradf=None):
    if gradf is None:
        x,_,_ = opt.fmin_l_bfgs_b(f, x0, fprime = np.gradient(f))
    else:
        x,_,_ = opt.fmin_l_bfgs_b(f, x0, fprime=gradf)
    return x

def compute_score( DVAL,v):
    w,b = v[0:-1], v[-1]
    S = (np.dot(vcol(w).T, DVAL) + b).ravel()
    return S


def logreg_obj_wrap_weighted(DTR, LTR, l, target_prior_pi):
    ZTR = 2* LTR -1
    nt = DTR[:, LTR == 1].shape[1]
    nf = DTR.shape[1] - nt
    def logreg_obj_weighted(v):
        w,b = v[0:-1], v[-1]
        S = (vcol(w).T @ DTR + b).ravel()
        EPS =np.where(ZTR == 1, target_prior_pi / nt, (1 - target_prior_pi) / nf)
        J = l / 2 * np.linalg.norm(w)**2 + np.sum( EPS * np.logaddexp(0, - ZTR * S) )
        return J
    return logreg_obj_weighted
    

def lr_classifier(DTR, LTR, DVAL, LVAL, l, pi):
    nt = DTR[:, LTR == 1].shape[1]
    nf = DTR.shape[1] - nt
    pi_emp = nt / (nt + nf)
    a_logreg = logreg_obj_wrap(DTR, LTR, l)
    v = optimize(a_logreg, np.zeros(DTR.shape[0]+1), None)
    scores = compute_score( DVAL, v)
    error_rate = np.sum( (scores > 0) != LVAL ) / float(LVAL.size)
    llrs = scores - np.log(pi_emp/(1-pi_emp))
    mindcf =dcf.compute_min_DCF(pi,1, 1 , llrs, LVAL)
    actdcf, _ = dcf.compute_DCF_normalized(pi, 1, 1, llrs , LVAL)
    print("----------------LOGISTIC REGRESSION Classifier----------------")
    print(f"lambda = {l}")
    print("Error rate = ", error_rate)
    print("MinDCF = %.4f, ActualDCF = %.4f\n" % (mindcf, actdcf))
    return llrs, mindcf, actdcf
    
def lr_classifier_weighted(DTR, LTR, DVAL, LVAL, l, pi):
    nt = DTR[:, LTR == 1].shape[1]
    nf = DTR.shape[1] - nt
    pi_emp = nt / (nt + nf)
    a_logreg = logreg_obj_wrap_weighted(DTR, LTR, l, pi)
    #grad = compute_gradient_weighted(DTR, LTR, l, nt, nf, pi)
    v = optimize(a_logreg, np.zeros(DTR.shape[0]+1))
    w,b = v[0:-1], v[-1]
    scores = compute_score( DVAL, v)
    error_rate = np.sum( (scores > 0) != LVAL ) / float(LVAL.size)
    llrs = scores - np.log(pi/(1-pi))
    mindcf =dcf.compute_min_DCF(pi,1, 1 , llrs, LVAL)
    actdcf, _ = dcf.compute_actual_DCF(pi, llrs  , LVAL)
    print("----------------WEIGHTED LOGISTIC REGRESSION Classifier----------------")
    print(f"lambda = {l}")
    print("Error rate = ", error_rate)
    print("MinDCF = %.4f, ActualDCF = %.4f\n" % (mindcf, actdcf))
    return llrs, mindcf, actdcf, w, b

def quadratic_lr_classifier(DTR, LTR, DTE, LTE, l, pi):
    def vec_x_xT (x):
        x = x[:, None]
        xxT = x.dot(x.T).reshape(x.size ** 2, order='F')
        return xxT

    dtr_e = np.apply_along_axis(vec_x_xT, 0, DTR)
    dte_e = np.apply_along_axis(vec_x_xT, 0, DTE)
    phi_r = np.array(np.vstack([dtr_e, DTR]))
    phi_e = np.array(np.vstack([dte_e, DTE]))
        
    x0 = np.zeros(phi_r.shape[0] + 1)
    nt = DTR[:, LTR == 1].shape[1]
    nf = DTR.shape[1] - nt
    pi_emp = nt / (nt + nf)
    a_logreg = logreg_obj_wrap_weighted(phi_r, LTR, l, 0.1)
    #grad = compute_gradient_weighted(phi_r, LTR, l, nt, nf, pi)
    v = optimize(a_logreg, x0)
    scores = compute_score( phi_e, v)
    error_rate = np.sum( (scores > 0) != LTE ) / float(LTE.size)
    llrs = scores - np.log(0.1/(1-0.1))
    mindcf =dcf.compute_min_DCF(pi,1, 1 , llrs, LTE)
    actdcf, _ = dcf.compute_DCF_normalized(pi,1,1, llrs , LTE, None)
    print("----------------QUADRATIC LOGISTIC REGRESSION Classifier----------------")
    print(f"lambda = {l}")
    print("Error rate = ", error_rate)
    print("MinDCF = %.4f, ActualDCF = %.4f\n" % (mindcf, actdcf))
    return llrs, mindcf, actdcf
    
# Used for k-fold
def trainWeightedLogRegBinary(DTR, LTR, l, pT):

    ZTR = LTR * 2.0 - 1.0 # We do it outside the objective function, since we only need to do it once
    
    wTar = pT / (ZTR>0).sum() # Compute the weights for the two classes
    wNon = (1-pT) / (ZTR<0).sum()

    def logreg_obj_with_grad(v): # We compute both the objective and its gradient to speed up the optimization
        w = v[:-1]
        b = v[-1]
        s = np.dot(vcol(w).T, DTR).ravel() + b

        loss = np.logaddexp(0, -ZTR * s)
        loss[ZTR>0] *= wTar # Apply the weights to the loss computations
        loss[ZTR<0] *= wNon

        G = -ZTR / (1.0 + np.exp(ZTR * s))
        G[ZTR > 0] *= wTar # Apply the weights to the gradient computations
        G[ZTR < 0] *= wNon
        
        GW = (vrow(G) * DTR).sum(1) + l * w.ravel()
        Gb = G.sum()
        return loss.sum() + l / 2 * np.linalg.norm(w)**2, np.hstack([GW, np.array(Gb)])

    vf =opt.fmin_l_bfgs_b(logreg_obj_with_grad, x0 = np.zeros(DTR.shape[0]+1))[0]
    #print ("Weighted Log-reg (pT %e) - lambda = %e - J*(w, b) = %e" % (pT, l, logreg_obj_with_grad(vf)[0]))
    return vf[:-1], vf[-1]