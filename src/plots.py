import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import generative
import utilities

FEATURE_NUMBER = 6


def draw_histgrams(D,L,m):
   D0 = D[:, L==0]
   D1 = D[:, L==1]
   for idx in range(m):
      plt.figure()
      plt.hist(D0[idx, :], bins=20, density=True, alpha=0.3,label="Counterfeit")
      plt.hist(D1[idx, :], bins=20, density=True, alpha=0.3,  label="Genuine")
      plt.xlabel(f"Feature number {idx}")   
      plt.legend()
      #plt.savefig('../images/hist_%d.jpg' % idx)
      #plt.close()
   plt.show()
   return

def draw_scatters(D,L,m):
   D0 = D[:, L==0]   
   D1 = D[:, L==1]
   already_plotted = set()
   
   for idx in range(m):
      for jdx in range(m):
         if idx == jdx:
            continue
         if (idx,jdx) not in already_plotted:
            already_plotted.add((jdx,idx))
         else:
            continue
         plt.figure()
         plt.scatter(D0[idx, :], D0[jdx, :], alpha=0.3, label="Counterfeit")
         plt.scatter(D1[idx, :], D1[jdx, :], alpha=0.3, label="Genuine")
         plt.xlabel(f"Feature number {idx}")
         plt.ylabel(f"Feature number {jdx}")
         plt.legend()
         #plt.savefig('../images/scatter_%d_%d.jpg' % (idx, jdx))
         plt.close()
   #plt.show()      
   return

def draw_one_histogram(D,L, title):
   D0 = D[:, L==0]
   D1 = D[:, L==1]
   plt.figure()
   plt.hist(D0[0, :], bins=10, density=True, alpha=0.3, label="Counterfeit")
   plt.hist(D1[0, :], bins=10, density=True, alpha=0.3, label="Genuine")
   plt.legend()
   plt.tight_layout()
   plt.title(title)
   #plt.close()
   plt.show()
   return

def draw_one_scatter(D,L, title):
   D0 = D[:, L==0]
   D1 = D[:, L==1]
   plt.figure()
   plt.scatter(D0[0, :], D0[1, :], alpha=0.3, label="Counterfeit")
   plt.scatter(D1[0, :], D1[1, :], alpha=0.3, label="Genuine")

   plt.legend()
   plt.title(title)
   plt.savefig(f"../images/scatter_{title}.jpg")
   #plt.close()
   plt.show()
   return

# Goal of lab 4
def draw_histogram_with_density(D, L):
   D0 = D[:, L==0]
   D1 = D[:, L==1]
   for i in range(FEATURE_NUMBER):
      mu0 = np.mean(D0[i:i+1, :], axis=1)
      mu1 = np.mean(D1[i:i+1, :], axis=1)
      C0 = utilities.compute_cov_matrix(D0[i:i+1, :])
      C1 = utilities.compute_cov_matrix(D1[i:i+1, :])
      x_range = np.linspace(np.min(D), np.max(D), 1000)
      density0 = np.exp(generative.logpdf_GAU_ND(x_range.reshape(1, -1), mu0, C0))
      density1 = np.exp(generative.logpdf_GAU_ND(x_range.reshape(1, -1), mu1, C1))
      plt.figure()
      plt.hist(D0[i,:], bins=20, density=True, alpha=0.3, label="Counterfeit")
      plt.hist(D1[i,:], bins=20, density=True, alpha=0.3, label="Genuine")
      plt.plot(x_range, density0, color="blue")
      plt.plot(x_range, density1, color = "orange")
      plt.title(f"Feature number {i}")
      plt.legend()
      #plt.savefig(f"../images/hist_density_{i}.jpg")
      plt.close()
   
def plot_correlation_matrix(corr0, corr1):
   plt.figure(figsize=(10, 8))
   sns.heatmap(corr0, annot=True, cmap='coolwarm', fmt=".4f", 
            xticklabels=range(corr0.shape[0]), 
            yticklabels=range(corr0.shape[0]))

   plt.title(f'Correlation Matrix for Class 0')
   plt.savefig('../images/corr0.jpg')
   
   plt.figure(figsize=(10, 8))
   sns.heatmap(corr1, annot=True, cmap='coolwarm', fmt=".4f", 
            xticklabels=range(corr1.shape[0]), 
            yticklabels=range(corr1.shape[0]))

   plt.title(f'Correlation Matrix for Class 1')
   plt.savefig('../images/corr1.jpg')
   
   plt.show()
   
def plot_roc_curve(pi, Cfn, Cfp, llrs, LTE):
   sorted_llrs = np.sort(llrs)
   FPR = []
   TPR = []
   for t in sorted_llrs:
      predictions = (llrs > t).astype(int)
      conf_matrix = np.zeros((2, 2))
      for i in range(len(LTE)):
         conf_matrix[predictions[i], LTE[i]] += 1
      Pfn = conf_matrix[0,1] / (conf_matrix[0,1] + conf_matrix[1, 1])
      Ptp = 1 - Pfn
      Pfp = conf_matrix[1,0] / (conf_matrix[1,0] + conf_matrix[0, 0])
      FPR.append(Pfp)
      TPR.append(Ptp)
        
   plt.figure()
   plt.grid()
   plt.title("ROC curve")
   plt.xlabel("False positive rate")
   plt.ylabel("True positive rate")
   plt.plot(FPR, TPR)
   plt.show()    
    
        
def plot_bayes_error(llrs, LTE):
   effPriorLogOdds = np.linspace(-4, 4,21)
   pi_tilde = 1 / (1 + np.exp( - effPriorLogOdds))
   Cfn = 1
   Cfp = 1
   dcf = []
   mnindcf = []
   for pi in pi_tilde:
      dcf.append(dcf.compute_DCF_normalized(pi, Cfn, Cfp, llrs, LTE))
      mnindcf.append(dcf.compute_min_DCF(pi, Cfn, Cfp, llrs, LTE))  
    
   plt.figure()
   plt.title("Bayes error")
   plt.plot(effPriorLogOdds, dcf, label="DCF", color="r")
   plt.plot(effPriorLogOdds, mnindcf, label="minDCF", color="b")
   plt.legend()
   plt.xlabel("prior log-odds")
   plt.ylabel("DCF value")
   plt.ylim(0, 1.1)
   plt.xlim(-3, 3)
   plt.show()
   
def plot_act_min_dft(x_value, act_dcf, min_dcf, name):
   plt.figure()
   plt.xscale("log", base=10)
   plt.title("Actual and Minimized DCF as function of " + name)
   plt.plot(x_value, act_dcf, 'o', label="Actual DCF",  color="r")
   plt.plot(x_value, min_dcf, 'x', label="Minimized DCF", color="b")
   plt.legend()
   plt.xlabel(name)
   plt.ylabel("DCF value")
   plt.show()