import numpy as np
import plots
import utilities
import classifier
import generative
import dcf
import matplotlib.pyplot as plt
import logreg
import svm
import gmm
import calibration

FILENAME = "data/trainData.txt"
EVAL_FILENAME = "data/evalData.txt"

FEATURE_NUMBER = 6
PCA_VALUES = [6, 5,4,3,2,1]

EFFECTIVE_PRIOR= [0.1, 0.5, 0.9]
APPLICATION_PRIOR = 0.1
KFOLD= 5

def calibrate_fusion_evaluate(DTR, LTR, DVAL, LVAL, DEVAL, LEVAL):
 #------------------------------ Bayes error plot of best 3 models ---------------------------------------------------
   plt.figure()
   qlr_llrs = logreg.quadratic_lr_classifier(DTR, LTR, DVAL, LVAL, 0.003162277660168379, APPLICATION_PRIOR)[0] 
   effectPriors, mindcf, actdcf = dcf.bayesPlot(qlr_llrs, LVAL, -4, 4, 21)
   plt.title("Best three models - Non calibrated scores")
   plt.plot(effectPriors, mindcf, '--', label="QUADR-LOG-REG - MinDCF")
   plt.plot(effectPriors, actdcf,  label="QUADR-LOG-REG - ActualDCF") 
   
   fscore = svm.train_dual_SVM_kernel(DTR, LTR, 31.622776601683793, svm.rbfKernel(0.1353352832366127))
   svm_llrs = fscore(DVAL)
   effectPriors, mindcf, actdcf = dcf.bayesPlot(svm_llrs, LVAL, -4, 4, 21)
   plt.plot(effectPriors, mindcf, '--', label="SVM-RBF - MinDCF")
   plt.plot(effectPriors, actdcf,  label="SVM-RBF - ActualDCF")
   covType = 'diagonal'
   numC0 = 8
   numC1 = 32
   print('Components: class0 = %d - class1 = %d' % (numC0, numC1))
   gmm0 = gmm.train_GMM_LBG_EM(DTR[:, LTR == 0], numC0, covType=covType, verbose=False, psiEig=0.01)
   gmm1 = gmm.train_GMM_LBG_EM(DTR[:, LTR == 1], numC1, covType=covType, verbose=False, psiEig=0.01)
   gmm_llrs = gmm.logpdf_GMM(DVAL, gmm1) - gmm.logpdf_GMM(DVAL, gmm0)
   effectPriors, mindcf, actdcf = dcf.bayesPlot(gmm_llrs, LVAL, -4, 4, 21)
   plt.plot(effectPriors, mindcf,'--',  label="GMM - MinDCF")
   plt.plot(effectPriors, actdcf,  label="GMM - ActualDCF")
   
   plt.legend()
   plt.xlabel("Prior log-odds")
   plt.ylabel("DCF")
   plt.savefig("images/best_three_models_uncalibrated.png")
   plt.close( )   
   
   #--------------------------------- Calibration  ---------------------------------------------------
   print("\nCalibration")
   plt.figure()
   plt.title("Best three models - Calibrated scores")
   plt.yticks(np.arange(0, 1.0, 0.2))
   plt.ylim(0, 1)
   print("Quadratic Logistic Regression calibrated dcf:")
   calibrated_scores_lr, calibrated_labels_lr = calibration.calibration_score(qlr_llrs, LVAL, 0.05)
   effectPriors, mindcf, actdcf = dcf.bayesPlot(calibrated_scores_lr, calibrated_labels_lr, -4, 4, 21)
   plt.plot(effectPriors, mindcf,'--',  label="QUADR-LOG-REG - MinDCF")
   plt.plot(effectPriors, actdcf,  label="QUADR-LOG-REG - ActualDCF")

   print()
   
   print("SVM RBF calibrated dcf:")
   calibrated_scores_svm, calibrated_labels_svm = calibration.calibration_score(svm_llrs, LVAL, 0.05)
   effectPriors, mindcf, actdcf = dcf.bayesPlot(calibrated_scores_svm, calibrated_labels_svm, -4, 4, 21)
   plt.plot(effectPriors, mindcf,'--',  label="SVM-RBF - MinDCF")
   plt.plot(effectPriors, actdcf,  label="SVM-RBF - ActualDCF")
   
   print()
   
   print("GMM calibrated dcf:")
   calibrated_scores_gmm, calibrated_labels_gmm = calibration.calibration_score(gmm_llrs, LVAL, 0.1)
   effectPriors, mindcf, actdcf = dcf.bayesPlot(calibrated_scores_gmm, calibrated_labels_gmm, -4, 4, 21)
   plt.plot(effectPriors, mindcf,'--',  label="GMM - MinDCF")
   plt.plot(effectPriors, actdcf,  label="GMM - ActualDCF")
   
   plt.legend()
   plt.xlabel("Prior log-odds")
   plt.ylabel("DCF")
   plt.savefig("images/best_three_models_calibrated.png")
   
   #-------------------------------------- Fusion ------------------------------------------------------
   
   fusedScores = []
   fusedLabels = []
   
   p = 0.05
   
   for foldIdx in range(KFOLD):
      SCAL_LR, SVAL_LR = utilities.extract_train_val_folds_from_ary(qlr_llrs, foldIdx)
      SCAL_SVM, SVAL_SVM = utilities.extract_train_val_folds_from_ary(svm_llrs, foldIdx)
      SCAL_GMM, SVAL_GMM = utilities.extract_train_val_folds_from_ary(gmm_llrs, foldIdx)
      LCAL, SVAL_labels = utilities.extract_train_val_folds_from_ary(LVAL, foldIdx)
      
      SCAL = np.vstack([SCAL_LR, SCAL_SVM, SCAL_GMM])
        # Train the model on the KFOLD - 1 training folds
      w, b = logreg.trainWeightedLogRegBinary(SCAL, LCAL, 0, p)
        # Build the validation scores "feature" matrix
      SVAL = np.vstack([SVAL_LR, SVAL_SVM, SVAL_GMM])
        # Apply the model to the validation fold
      calibrated_SVAL = (w.T @ SVAL + b - np.log(p / (1 - p))).ravel()
      fusedScores.append(calibrated_SVAL)
      fusedLabels.append(SVAL_labels)
      
   fusedScores = np.hstack(fusedScores)
   fusedLabels = np.hstack(fusedLabels)
   
   print('Fusion')
   print('\tValidation set')
   print('\t\tminDCF(p=0.1)         : %.4f' % dcf.compute_min_DCF(APPLICATION_PRIOR, 1, 1, fusedScores, fusedLabels))  # Calibration may change minDCF due to being fold-dependent (thus it's not globally affine anymore)
   print('\t\tactDCF(p=0.1)         : %.4f' % dcf.compute_actual_DCF(APPLICATION_PRIOR, fusedScores, fusedLabels)[0])
   
   plt.figure()
   plt.xlabel("prior log-odds")
   plt.ylabel("DCF value")
   plt.title("Fusion - Calibrated scores")
   effectPriors, mindcf, actdcf = dcf.bayesPlot(fusedScores, fusedLabels, -4, 4, 21)
   plt.plot(effectPriors, mindcf, '--', label="MinDCF - Fusion")
   plt.plot(effectPriors, actdcf, label="ActualDCF - Fusion")
   effectPriors, mindcf, actdcf = dcf.bayesPlot(calibrated_scores_lr, calibrated_labels_lr, -4, 4, 21)
   plt.plot(effectPriors, actdcf, label="ActualDCF - Quadratic Logistic Regression")
   effectPriors, mindcf, actdcf = dcf.bayesPlot(calibrated_scores_svm, calibrated_labels_svm, -4, 4, 21)
   plt.plot(effectPriors, actdcf, label="ActualDCF - SVM RBF")
   effectPriors, mindcf, actdcf = dcf.bayesPlot(calibrated_scores_gmm, calibrated_labels_gmm, -4, 4, 21)
   plt.plot(effectPriors, actdcf, label="ActualDCF - GMM")
   plt.legend()
   plt.yticks(np.arange(0, 1.0, 0.2))
   plt.ylim(0, 1)
   plt.savefig("images/fusion_calibrated.png")
   plt.close()
   
   #-------------------------------------- Evaluation ------------------------------------------------------
   
   # selected final best system for target application:
   # GMM: Diagonal Covariance GMM - numC0 = 8 - numC1 = 32
   # minDCF: 0.1312
   # actDCF: 0.1517
   print("Initial evaluation for best GMM model")
   eval_gmm_llrs = gmm.logpdf_GMM(DEVAL, gmm1) - gmm.logpdf_GMM(DEVAL, gmm0)
   w,b = logreg.trainWeightedLogRegBinary(utilities.vrow(eval_gmm_llrs), LEVAL, 0, APPLICATION_PRIOR)
   calibrated_eval_gmm_llrs = (w.T @ utilities.vrow(eval_gmm_llrs) + b - np.log(APPLICATION_PRIOR / (1 - APPLICATION_PRIOR))).ravel()
   eval_gmm_predictions = dcf.compute_optimal_Bayes_binary_llr(eval_gmm_llrs, APPLICATION_PRIOR, 1, 1)
   err = (eval_gmm_predictions != LEVAL).sum() / float(LEVAL.size)
   
   print('\tEvaluation set GMM')
   print('\t\tError rate: %.1f' % (err * 100))
   print('\t\tminDCF(p=0.1)         : %.4f' % dcf.compute_min_DCF(APPLICATION_PRIOR, 1, 1, eval_gmm_llrs, LEVAL))
   print('\t\tactDCF(p=0.1), no cal.: %.4f' % dcf.compute_actual_DCF(APPLICATION_PRIOR, eval_gmm_llrs, LEVAL)[0])
   print('\t\tactDCF(p=0.1), cal.   : %.4f' % dcf.compute_actual_DCF(APPLICATION_PRIOR, calibrated_eval_gmm_llrs, LEVAL)[0])

   plt.figure()
   plt.title("Evaluation GMM")
   plt.xlabel("prior log-odds")
   plt.ylabel("DCF value")
   effectPriors, mindcf_nocal_gmm, actdcf_nocal_gmm = dcf.bayesPlot(eval_gmm_llrs, LEVAL, -4, 4, 21)
   effectPriors, mindcf_cal_gmm, actdcf_cal_gmm = dcf.bayesPlot(calibrated_eval_gmm_llrs, LEVAL, -4, 4, 21)
   plt.plot(effectPriors, mindcf_nocal_gmm, '--', label="MinDCF - GMM")
   plt.plot(effectPriors, actdcf_nocal_gmm, label="ActualDCF - GMM")
   plt.plot(effectPriors, actdcf_cal_gmm, label="ActualDCF - GMM calibrated")
   plt.ylim(0, 1)
   plt.legend()
   plt.savefig("images/evaluation_gmm.png")
   
   # Quadratic Logistic Regression Evaluation
   eval_qlr_llrs = logreg.quadratic_lr_classifier(DTR, LTR, DEVAL, LEVAL,0.003162277660168379, APPLICATION_PRIOR)[0]
   w,b = logreg.trainWeightedLogRegBinary(utilities.vrow(eval_qlr_llrs), LEVAL , 0, 0.05)
   calibrated_eval_gmm_llrs = (w.T @ utilities.vrow(eval_qlr_llrs) + b - np.log(APPLICATION_PRIOR / (1 - APPLICATION_PRIOR))).ravel()
   print("\tEvaluation set QLR")
   print('\t\tminDCF(p=0.1)         : %.4f' % dcf.compute_min_DCF(APPLICATION_PRIOR, 1, 1, eval_qlr_llrs, LEVAL))
   print('\t\tactDCF(p=0.1), no cal.: %.4f' % dcf.compute_actual_DCF(APPLICATION_PRIOR, eval_qlr_llrs, LEVAL)[0])
   print('\t\tactDCF(p=0.1), cal.   : %.4f' % dcf.compute_actual_DCF(APPLICATION_PRIOR, calibrated_eval_gmm_llrs, LEVAL)[0])
   
   logOdds, mindcf_nocal_qlr, actdcf_nocal_qlr = dcf.bayesPlot(eval_qlr_llrs, LEVAL, -4, 4, 21)
   logOdds, mindcf_cal_qlr, actdcf_cal_qlr = dcf.bayesPlot(calibrated_eval_gmm_llrs, LEVAL, -4, 4, 21)
   
   # SVM evaluation
   
   eval_svm_llrs = fscore(DEVAL)
   w,b = logreg.trainWeightedLogRegBinary(utilities.vrow(eval_svm_llrs), LEVAL, 0, 0.05)
   calibrated_eval_svm_llrs = (w.T @ utilities.vrow(eval_svm_llrs) + b - np.log(APPLICATION_PRIOR / (1 - APPLICATION_PRIOR))).ravel()
   
   print("\tEvaluation set SVM")
   print('\t\tminDCF(p=0.1)         : %.4f' % dcf.compute_min_DCF(APPLICATION_PRIOR, 1, 1, eval_svm_llrs, LEVAL))
   print('\t\tactDCF(p=0.1), no cal.: %.4f' % dcf.compute_actual_DCF(APPLICATION_PRIOR, eval_svm_llrs, LEVAL)[0])
   print('\t\tactDCF(p=0.1), cal.   : %.4f' % dcf.compute_actual_DCF(APPLICATION_PRIOR, calibrated_eval_svm_llrs, LEVAL)[0])
   
   logOdds, mindcf_nocal_svm, actdcf_nocal_svm = dcf.bayesPlot(eval_svm_llrs, LEVAL, -4, 4, 21)
   logOdds, mindcf_cal_svm, actdcf_cal_svm = dcf.bayesPlot(calibrated_eval_svm_llrs, LEVAL, -4, 4, 21)
   
   # Fusion evaluation
   
   SMatrix = np.vstack([qlr_llrs, svm_llrs, gmm_llrs])
   w,b = logreg.trainWeightedLogRegBinary(SMatrix, LVAL, 0, 0.05)
   
   eval_SMatrix = np.vstack([eval_qlr_llrs, eval_svm_llrs, eval_gmm_llrs])
   eval_fused_scores = (w.T @ eval_SMatrix + b - np.log(APPLICATION_PRIOR / (1 - APPLICATION_PRIOR))).ravel()
   print("Evaluation set fusion")
   print('\t\tminDCF(p=0.1)         : %.4f' % dcf.compute_min_DCF(APPLICATION_PRIOR, 1, 1, eval_fused_scores, LEVAL))
   print('\t\tactDCF(p=0.1), no cal.: %.4f' % dcf.compute_actual_DCF(APPLICATION_PRIOR, eval_fused_scores, LEVAL)[0])
   
   logOdds, mindcf_cal_fusion, actdcf_cal_fusion = dcf.bayesPlot(eval_fused_scores, LEVAL, -4, 4, 21)
   
   plt.figure()
   plt.title("ActualDCF - Evaluation of best models")
   plt.xlabel("prior log-odds")
   plt.ylabel("DCF value")
   plt.plot(logOdds, actdcf_cal_qlr, label="ActualDCF - QLR")
   plt.plot(logOdds, actdcf_cal_svm, label="ActualDCF - SVM")
   plt.plot(logOdds, actdcf_cal_gmm, label="ActualDCF - GMM")
   plt.plot(logOdds, actdcf_cal_fusion, label="ActualDCF - Fusion")
   plt.ylim(0, 1)
   plt.legend()
   plt.savefig("images/evaluation_actual_dcf.png")
   plt.close()
   
   plt.figure()
   plt.title("Evaluation of QLR")
   plt.xlabel("prior log-odds")
   plt.ylabel("DCF value")
   plt.plot(logOdds, mindcf_nocal_qlr, '--', label="MinDCF - QLR")
   plt.plot(logOdds, actdcf_nocal_qlr, label="ActualDCF - QLR")
   plt.plot(logOdds, actdcf_cal_qlr, label="ActualDCF - QLR calibrated")
   plt.ylim(0, 1)
   plt.legend()
   plt.savefig("images/evaluation_qlr.png")
   
   plt.figure()
   plt.title("Evaluation of SVM")
   plt.xlabel("prior log-odds")
   plt.ylabel("DCF value")
   plt.plot(logOdds, mindcf_nocal_svm, '--', label="MinDCF - SVM")
   plt.plot(logOdds, actdcf_nocal_svm, label="ActualDCF - SVM")
   plt.plot(logOdds, actdcf_cal_svm, label="ActualDCF - SVM calibrated")
   plt.ylim(0, 1)
   plt.legend()
   plt.savefig("images/evaluation_svm.png")
   plt.close()
   
   # print("Full Covariance GMM")
   # covType = 'full'
   # print(covType)

   # gmm0_list = []
   # gmm1_list = []
   # for numC in [1, 2, 4, 8, 16, 32]:
   #    gmm0 = gmm.train_GMM_LBG_EM(DTR[:, LTR == 0], numC, covType=covType, verbose=False, psiEig=0.01)
   #    gmm0_list.append(gmm0)
   #    gmm1 = gmm.train_GMM_LBG_EM(DTR[:, LTR == 1], numC, covType=covType, verbose=False, psiEig=0.01)
   #    gmm1_list.append(gmm1)

   # for i in range(len(gmm0_list)):
   #    for j in range(len(gmm1_list)):
   #       print()
   #       SLLR = gmm.logpdf_GMM(DEVAL, gmm1_list[j]) - gmm.logpdf_GMM(DEVAL, gmm0_list[i])
   #       print('numC0 = %d - numC1 = %d' % (2**i, 2**j))
   #       print('minDCF - pT = 0.1: %.4f' % dcf.compute_min_DCF(APPLICATION_PRIOR, 1, 1, SLLR, LEVAL))
   # print()

   # print("Diagonal Covariance GMM")
   # covType = 'diagonal'
   # print(covType)

   # gmm0_list = []
   # gmm1_list = []
   # for numC in [1, 2, 4, 8, 16, 32]:
   #    gmm0 = gmm.train_GMM_LBG_EM(DTR[:, LTR == 0], numC, covType=covType, verbose=False, psiEig=0.01)
   #    gmm0_list.append(gmm0)
   #    gmm1 = gmm.train_GMM_LBG_EM(DTR[:, LTR == 1], numC, covType=covType, verbose=False, psiEig=0.01)
   #    gmm1_list.append(gmm1)

   # for i in range(len(gmm0_list)):
   #    for j in range(len(gmm1_list)):
   #       print()
   #       SLLR = gmm.logpdf_GMM(DEVAL, gmm1_list[j]) - gmm.logpdf_GMM(DEVAL, gmm0_list[i])
   #       print('numC0 = %d - numC1 = %d' % (2**i, 2**j))
   #       print('minDCF - pT = 0.1: %.4f' % dcf.compute_min_DCF(APPLICATION_PRIOR, 1, 1, SLLR, LEVAL))
   # print()


def train(DTR, LTR, DVAL, EVAL):
     
   #------------------------------------------- Gaussian classifier------------------------------------------------
   
   for pca_value in PCA_VALUES:
      print("***********************PCA value: ", pca_value, "************************")    
      U_pca = utilities.pca(DTR, pca_value)
      DTR_pca = np.dot(U_pca.T, DTR)
      DTE_pca = np.dot(U_pca.T, DVAL)          

      for prior in EFFECTIVE_PRIOR:
         dcf.evaluate("MVG", (prior, 1,1), generative.mvg_classifier(DTR_pca, LTR, DTE_pca, LVAL), LVAL)
         dcf.evaluate("Tied Covariance Gaussian", (prior, 1,1), generative.tied_covariance_gaussian_classifier(DTR_pca, LTR, DTE_pca, LVAL), LVAL)
         dcf.evaluate("Naive Bayes", (prior, 1,1), generative.naive_bayes_classifier(DTR_pca, LTR, DTE_pca, LVAL), LVAL)
      
   #-------------------------------------- Logistic regression classifier-------------------------------------------
   
   DTR = utilities.center_data(DTR)
   DVAL = utilities.center_data(DVAL)
   lambda_values = np.logspace(-4, 2, 13)
   mindcf_array = []
   actdcf_array = []

   for l in lambda_values:
      llrs, mindcf, actdcf = logreg.lr_classifier(DTR, LTR, DVAL, LVAL, l, APPLICATION_PRIOR)
      mindcf_array.append(mindcf)
      actdcf_array.append(actdcf)

   mindcf_array = np.array(mindcf_array)
   actdcf_array = np.array(actdcf_array)

   plt.figure()
   plt.xscale("log", base=10)
   plt.plot(lambda_values, mindcf_array, 'o', label="MinDCF")
   plt.plot(lambda_values, actdcf_array, 'x' , label="DCF")
   plt.xlabel('Lambda')
   plt.ylabel('DCF')
   plt.legend()
   plt.show()
   
   #-------------------------------------- Quadratic Logistic regression classifier-------------------------------------------
   
   lambda_values = np.logspace(-4, 2, 13)
   mindcf_array = []
   actdcf_array = []
   
   for l in lambda_values:
      llrs, mindcf, actdcf = logreg.quadratic_lr_classifier(DTR, LTR, DVAL, LVAL, l, APPLICATION_PRIOR)
      mindcf_array.append(mindcf)
      actdcf_array.append(actdcf)
      
   mindcf_array = np.array(mindcf_array)
   actdcf_array = np.array(actdcf_array)
   
   plt.figure()
   plt.xscale("log", base=10)
   plt.title("Quadratic Logistic Regression")
   plt.plot(lambda_values, mindcf_array, 'o', label="MinDCF")
   plt.plot(lambda_values, actdcf_array, 'x' , label="DCF")
   plt.xlabel('Lambda')
   plt.ylabel('DCF')
   plt.legend()
   plt.show()
   
   #---------------------------------- Support Vector Machine Linear classifier -------------------------------------
   
   C_values = np.logspace(-5,0,11)
   mindcf_array = []
   actdcf_array = []
   # DTR = utilities.center_data(DTR)
   # DVAL = utilities.center_data(DVAL)
   
   for c in C_values:
      w,b = svm.train_dual_SVM_linear(DTR, LTR, c, 1)
      SVAL = (utilities.vrow(w) @ DVAL + b).ravel()
      PVAL = (SVAL > 0) * 1
      err = (PVAL != LVAL).sum() / float(LVAL.size)
      mindcf = dcf.compute_min_DCF(APPLICATION_PRIOR, 1, 1, SVAL, LVAL)
      actdcf, _ = dcf.compute_actual_DCF(APPLICATION_PRIOR, SVAL, LVAL)
      mindcf_array.append(mindcf)
      actdcf_array.append(actdcf)
      print("Error rate: ", err)
      print("C: ", c)
      print("MinDCF: %.4f" % mindcf)
      print("ActualDCF: %.4f " % actdcf)
      print()
      
   mindcf_array = np.array(mindcf_array)
   actdcf_array = np.array(actdcf_array)
   
   plots.plot_act_min_dft(C_values, actdcf_array, mindcf_array, "C")

   #----------------------------------Support Vector Machine Polynomial Kernel classifier ---------------------------------------
   
   C_values = np.logspace(-5,0,11)
   mindcf_array = []
   actdcf_array = []
   
   for c in C_values:
      fscore = svm.train_dual_SVM_kernel(DTR, LTR, c, svm.polyKernel(2,1))
      SVAL = fscore(DVAL)
      PVAL = (SVAL > 0) * 1
      err = (PVAL != LVAL).sum() / float(LVAL.size)
      mindcf_array.append(dcf.compute_min_DCF(APPLICATION_PRIOR, 1, 1, SVAL, LVAL))
      actdcf, _ = dcf.compute_actual_DCF(APPLICATION_PRIOR, SVAL, LVAL)
      actdcf_array.append(actdcf)
      print("Error rate: ", err)
      print("C: ", c)
      print("MinDCF: %.4f" % mindcf_array[-1])
      print("ActualDCF: %.4f" % actdcf)
      print()
   mindcf_array = np.array(mindcf_array)
   actdcf_array = np.array(actdcf_array)
   
   plots.plot_act_min_dft(C_values, actdcf_array, mindcf_array, "C - Polynomial Kernel")
   
   #----------------------------------Support Vector Machine RBF Kernel classifier ---------------------------------------
   
   C_values = np.logspace(-3,2 ,11)
   gamma_values = np.array( [np.exp(-4), np.exp(-3), np.exp(-2), np.exp(-1)] ) 
   plt.figure()
   plt.xscale("log", base=10)
   plt.title("SVM RBF Kernel")
   for gamma in gamma_values:
      mindcf_array = []
      actdcf_array = []
      for c in C_values:
         fscore = svm.train_dual_SVM_kernel(DTR, LTR, c, svm.rbfKernel(gamma))
         SVAL = fscore(DVAL)
         PVAL = (SVAL > 0) * 1
         err = (PVAL != LVAL).sum() / float(LVAL.size)
         mindcf_array.append(dcf.compute_min_DCF(APPLICATION_PRIOR, 1, 1, SVAL, LVAL))
         actdcf, _ = dcf.compute_actual_DCF(APPLICATION_PRIOR, SVAL, LVAL)
         actdcf_array.append(actdcf)
         print("Error rate: ", err)
         print("C: ", c)
         print("Gamma: ", gamma)
         print("MinDCF: %.4f" % mindcf_array[-1])
         print("ActualDCF: %.4f" % actdcf)
         print()
      mindcf_array = np.array(mindcf_array)
      actdcf_array = np.array(actdcf_array)
      
      plt.plot(C_values, mindcf_array, 'o-', label=f"MinDCF - gamma={gamma}")
      plt.plot(C_values, actdcf_array, 'x-', label=f"ActualDCF - gamma={gamma}")
   plt.legend()
   plt.xlabel("C")
   plt.ylabel("DCF")
   plt.show()
   
   #--------------------------------- Gaussian Mixture Model classifier -------------------------------------------
   
   gmm.GMM_train_classification(DTR, LTR, DVAL, LVAL, APPLICATION_PRIOR, 'full')
   
   gmm.GMM_train_classification(DTR, LTR, DVAL, LVAL, APPLICATION_PRIOR, 'diagonal')
   
   return

if __name__=="__main__":
   
   D,L = utilities.load(FILENAME)
   (DTR, LTR), (DVAL, LVAL) = utilities.split_db_2to1(D, L)
   DEVAL, LEVAL = utilities.load(EVAL_FILENAME)
   #train(DTR, LTR, DVAL, LVAL)
   #calibrate_fusion_evaluate(DTR, LTR, DVAL, LVAL, DEVAL, LEVAL)


   