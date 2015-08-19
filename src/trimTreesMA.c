/*******************************************************************
   Copyright (C) 2014 Yael Grushka-Cockayne, Victor Richmond R. Jose, 
   and Kenneth C. Lichtendahl Jr.

   This program is free software; you can redistribute it and/or
   modify it under the terms of the GNU General Public License
   as published by the Free Software Foundation; either version 2
   of the License, or (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.
*******************************************************************/

#include <Rmath.h>
#include <R.h>
#include <math.h>

void zeroIntMA(int *x, int length) {
    memset(x, 0, length * sizeof(int));
}

void zeroDoubleMA(double *x, int length) {
    memset(x, 0, length * sizeof(double));
}

void trimTreesMA(
          /* random forest inputs. */
          int *inbagCountSorted,
          int *termNodetrainSorted,
          int *ntree,
          double *ytrainSorted,
          int *ntrain,
          double *forestSupport,
          int *nSupport,
          int *termNodeNewX,
          double *ytest,
          int *ntest,
 
          /* user inputs. */
          double *trim,
          Rboolean *trimIsExterior,
          double *uQuantiles,
          int *nQuantiles,
          
          /* tree outputs. */
          double *treeValues,
          double *treeCounts,
          double *treeCumCounts,
          double *treeCDFs,
          double *treePMFs,
          double *treeMeans,
          double *treeVars,
          double *treePITs,
          double *treeQuantiles,
          double *treeFirstPMFValues,
          
          /* ensembles outputs. */
          double *bracketingRate,
          double *bracketingRateAllPairs,
                    
          double *trimmedEnsembleCDFs,
          double *trimmedEnsemblePMFs,
          double *trimmedEnsembleMeans,
          double *trimmedEnsembleVars,
          double *trimmedEnsemblePITs,
          double *trimmedEnsembleQuantiles,
          double *trimmedEnsembleComponentScores,
          double *trimmedEnsembleScores,
          
          double *untrimmedEnsembleCDFs,
          double *untrimmedEnsemblePMFs,
          double *untrimmedEnsembleMeans,
          double *untrimmedEnsembleVars,
          double *untrimmedEnsemblePITs,
          double *untrimmedEnsembleQuantiles,
          double *untrimmedEnsembleComponentScores,
          double *untrimmedEnsembleScores,
          
          double *tol
          ) 
  
  {

  int t, i, j, k, lo, hi, nTrim, *index, *index2, indexPIT;
  double *cdfValuesToTrim, *meansToSort, trimmedSum;
  
  index = (int *) R_alloc(*nQuantiles, sizeof(int));
  index2 = (int *) R_alloc(*ntree, sizeof(int));
  cdfValuesToTrim = (double *) R_alloc(*ntree, sizeof(double));
  meansToSort = (double *) R_alloc(*ntree, sizeof(double));
 
  /* Set the low and high indices for trimming. */
  if(*trimIsExterior) {
    lo = (int)((*ntree) * (*trim));
    hi = *ntree - lo;
    if(lo == hi) { /* lo == hi when exterior trimming level is 0.5 and ntree is even. 
        In this case, the trimmed ensemble is the median forecast. */
      lo -= 1; 
      hi += 1; 
    }
    nTrim = *ntree - 2 * lo;
  } else {
    lo = (int)((0.5 - (*trim)) * (*ntree));
    if(lo == 0) lo += 1;
    hi = (*ntree) - lo + 1;
    nTrim = 2 * lo;
  }  

  /* Start big loop over the rows in the test set. */
  for(t = 0; t <= *ntest - 1; t++) {
      
    /* ------------
       TREE OUTPUTS
       ------------ */

    /* This loop finds each tree's y values (not necessarily unique) that are both inbag and in the new X's terminal node.
       (Note that the y values in a training set may not be unique.)  
       This loop also finds each tree's counts and cumulative counts of these y values and listed them by the unique y values. */
    zeroDoubleMA(treeValues, (*ntrain) * (*ntree));
    zeroDoubleMA(treeCounts, (*nSupport) * (*ntree));
    zeroDoubleMA(treeCumCounts, (*nSupport + 1) * (*ntree));
    zeroDoubleMA(treeCDFs, (*nSupport + 1) * (*ntree));
    zeroDoubleMA(treePMFs, (*nSupport) * (*ntree));
    for(i = 0; i <= *ntree - 1; i++) {
      k = 0;
      for(j = 0; j <= *nSupport - 1; j++) {
        while(fabs(forestSupport[j] - ytrainSorted[k]) < *tol) {
          if(termNodetrainSorted[i * (*ntrain) + k] == termNodeNewX[i * (*ntest) + t] && 
             inbagCountSorted[i * (*ntrain) + k] != 0) {
            treeValues[i * (*ntrain) + k] =  ytrainSorted[k];     
            treeCounts[i * (*nSupport) + j] += inbagCountSorted[i * (*ntrain) + k];
          } else treeValues[i * (*ntrain) + k] =  NA_REAL;
          k++;  
          if(k > *ntrain - 1) break;
        }
        treeCumCounts[i * (*nSupport + 1) + j + 1] = treeCumCounts[i * (*nSupport + 1) + j] + treeCounts[i * (*nSupport) + j];      
      }
    }
  
    /* This loop finds the trees' cdfs. */
    for(j = 0; j <= *nSupport - 1; j++) {
      for(i = 0; i <= *ntree - 1; i++) {
        treeCDFs[i * (*nSupport + 1) + j + 1] =  treeCumCounts[i * (*nSupport + 1) + j + 1] / treeCumCounts[i * (*nSupport  + 1) + *nSupport];
        treePMFs[i * (*nSupport) + j] =  treeCounts[i * (*nSupport) + j] / treeCumCounts[i * (*nSupport  + 1) + *nSupport];
        if(j == 0) treeFirstPMFValues[i * (*ntest) + t] = treePMFs[i * (*nSupport)];
      }
    }
            
    /*This loop calculates the trees' means and variances. */
    for(i = 0; i <= *ntree - 1; i++) {
      for(j = 0; j <= *nSupport - 1; j++) {
        treeMeans[i * (*ntest) + t] +=  treePMFs[i * (*nSupport) + j] * forestSupport[j]; 
        treeVars[i * (*ntest) + t] +=  treePMFs[i * (*nSupport) + j] * forestSupport[j] * forestSupport[j];
      }
       treeVars[i * (*ntest) + t] -= (treeMeans[i * (*ntest) + t] * treeMeans[i * (*ntest) + t]);
    }
    
    /* This loop finds each tree's PIT.  */
    for(i = 0; i <= *ntree - 1; i++) {
        if(ytest[t] < forestSupport[0]) {
          treePITs[i * (*ntest) + t] = 0;
        } else {
        indexPIT = *nSupport - 1;
        while(ytest[t] < forestSupport[indexPIT])  indexPIT -= 1;
        treePITs[i * (*ntest) + t] = treeCDFs[i * (*nSupport + 1) + indexPIT + 1];
        }
    }
    
    /* This loop finds the quantiles of each tree's cdf.  */  
    zeroDoubleMA(treeQuantiles, (*ntree) * (*nQuantiles));
    for(i = 0; i <= *ntree - 1; i++) {
      zeroIntMA(index, *nQuantiles);
      for(k = *nQuantiles - 1; k >= 0; k--) {
        if(k == *nQuantiles - 1) index[k] = *nSupport - 1;
        else index[k] = index[k + 1];
        /* while condition is u <= cdf value */
        while((uQuantiles[k] < treeCDFs[i * (*nSupport + 1) + index[k]]) ||   
              (fabs(uQuantiles[k] - treeCDFs[i * (*nSupport + 1) + index[k]]) < *tol)
             ) index[k] -= 1;
        treeQuantiles[k * (*ntree) + i] = forestSupport[index[k]];
        index[k]++;
      }
    }
    
    /* This loop calculates the bracketing rate among the trees, for each test value.  */  
    double countAbove;
    countAbove = 0;
    for(i = 0; i <= *ntree - 1; i++) {
      if(treeMeans[i * (*ntest) + t] > ytest[t]) countAbove ++;
    }  
    bracketingRate[t] = 2* countAbove/ (double)(*ntree) *(1 - countAbove/ (double)(*ntree));
   
    /* This loop calculates the bracketing rate among each pair of trees, for each test value.  */
    for(i = 0; i <= *ntree - 2; i++){
      for(j = i + 1; j <= *ntree - 1; j++){
        if((treeMeans[i * (*ntest) + t] <= ytest[t]) && (ytest[t] <= treeMeans[j * (*ntest) + t]))
          bracketingRateAllPairs [j * (*ntree) + i] ++;
        if((treeMeans[i * (*ntest) + t] >= ytest[t]) && (ytest[t] >= treeMeans[j * (*ntest) + t]))
          bracketingRateAllPairs [j * (*ntree) + i] ++;
      }
    } 

    /* --------------------------
       UNTRIMMED ENSEMBLE OUTPUTS
       -------------------------- */

    /* This loop finds the untrimmed ensemble's cdf. */
    for(j = 0; j <= *nSupport - 1; j++) {
      for(i = 0; i <= *ntree - 1; i++) {
      untrimmedEnsembleCDFs[(j + 1) * (*ntest) +  t] += treeCDFs[i * (*nSupport + 1) + j +  1] / (double)(*ntree);
      }
      untrimmedEnsemblePMFs[j * (*ntest) +  t] = untrimmedEnsembleCDFs[(j + 1) * (*ntest) +  t] - untrimmedEnsembleCDFs[j * (*ntest) +  t];
    }  
    
    /* This loop finds the untrimmed ensemble's mean and variance. */
    for(i = 0; i <= *ntree - 1; i++) {
      untrimmedEnsembleMeans[t] += treeMeans[i * (*ntest) + t] / (double)(*ntree);
    }
    
    
    for(i = 0; i <= *ntree - 1; i++) {
      untrimmedEnsembleVars[t] += (treeMeans[i * (*ntest) + t] - untrimmedEnsembleMeans[t]) 
                                      * (treeMeans[i * (*ntest) + t] - untrimmedEnsembleMeans[t]) / (double)(*ntree)
                                      + treeVars[i * (*ntest) + t] / (double)(*ntree);
    }
       
    /* This statement finds the untrimmed ensemble's PIT. */
   if(ytest[t] < forestSupport[0]) {
      untrimmedEnsemblePITs[t] = 0;
    } else {
      indexPIT = *nSupport - 1;
      while(ytest[t] < forestSupport[indexPIT])  indexPIT -= 1;
      untrimmedEnsemblePITs[t] = untrimmedEnsembleCDFs[(indexPIT + 1) * (*ntest) +  t];
    }    
    
    /* This loop finds the untrimmed ensemble's quantiles.  */
    zeroIntMA(index, *nQuantiles);
    zeroDoubleMA(untrimmedEnsembleQuantiles, *nQuantiles);
    for(k = *nQuantiles - 1; k >= 0; k--) {
      if(k == *nQuantiles - 1) index[k] = *nSupport - 1;
      else index[k] = index[k + 1];
      /* while condition is u <= cdf value */
      while((uQuantiles[k] < untrimmedEnsembleCDFs[index[k] * (*ntest) + t]) ||   
            (fabs(uQuantiles[k] - untrimmedEnsembleCDFs[index[k] * (*ntest) + t]) < *tol)
           ) index[k] -= 1;
      untrimmedEnsembleQuantiles[k] = forestSupport[index[k]];
      index[k]++;
    }  

    /* These loops find the untrimmed ensemble's scores: LinQuanS, LogQuanS, RPS and TMS. */
    zeroDoubleMA(untrimmedEnsembleComponentScores, *nQuantiles*2);
    for(k = 0; k <= *nQuantiles - 1; k++) {
      if(untrimmedEnsembleQuantiles[k] <= ytest[t]) 
        untrimmedEnsembleComponentScores[k] = - uQuantiles[k] * (ytest[t] - untrimmedEnsembleQuantiles[k]);
      else 
        untrimmedEnsembleComponentScores[k] = - (1 - uQuantiles[k]) * (untrimmedEnsembleQuantiles[k] - ytest[t]);
      untrimmedEnsembleScores[t] += untrimmedEnsembleComponentScores[k];   
      }
    
    for(k = 0; k <= *nQuantiles - 1; k++) {
      if(untrimmedEnsembleQuantiles[k] <= ytest[t]) 
        untrimmedEnsembleComponentScores[*nQuantiles + k] = - uQuantiles[k] * (log(ytest[t]) - log(untrimmedEnsembleQuantiles[k]));
      else 
        untrimmedEnsembleComponentScores[*nQuantiles + k] = - (1 - uQuantiles[k]) * (log(untrimmedEnsembleQuantiles[k]) - log(ytest[t]));
      untrimmedEnsembleScores[(*ntest) + t] += untrimmedEnsembleComponentScores[*nQuantiles + k];   
      }
    
    for(j = 0; j <= *nSupport - 2; j++) {
      if(forestSupport[j] < ytest[t])
        untrimmedEnsembleScores[2 * (*ntest) + t] -= untrimmedEnsembleCDFs[(j + 1) * (*ntest) +  t] * untrimmedEnsembleCDFs[j * (*ntest) +  t];
      else 
        untrimmedEnsembleScores[2 * (*ntest) + t] -= (1 - untrimmedEnsembleCDFs[(j + 1) * (*ntest) +  t]) * (1 - untrimmedEnsembleCDFs[(j + 1) * (*ntest) +  t]);
      }    
    
    untrimmedEnsembleScores[3 * (*ntest) + t] = log(1/untrimmedEnsembleVars[t]) - (1/untrimmedEnsembleVars[t]) * (ytest[t]-untrimmedEnsembleMeans[t]) * (ytest[t] - untrimmedEnsembleMeans[t]);    
   

    /* ------------------------
       TRIMMED ENSEMBLE OUTPUTS
       ------------------------ */  
    /* This loop finds the trimmed ensemble's cdf. */
    zeroDoubleMA(cdfValuesToTrim, *ntree);
    zeroDoubleMA(meansToSort, *ntree);
    zeroIntMA(index2, *ntree);
    if(fabs(*trim) < *tol){
      for(j = 0; j <= *nSupport - 1; j++) {
        trimmedEnsembleCDFs[(j + 1) * (*ntest) +  t] = untrimmedEnsembleCDFs[(j  + 1) * (*ntest) +  t];
        trimmedEnsemblePMFs[j * (*ntest) +  t] = trimmedEnsembleCDFs[(j + 1) * (*ntest) +  t] - trimmedEnsembleCDFs[j * (*ntest) +  t];
      }
    }
    else {
      for(i = 0; i <= *ntree - 1; i++) {
        meansToSort[i] = treeMeans[i * (*ntest) + t];
        index2[i] = i;
        }
      rsort_with_index(meansToSort, index2, *ntree);
      for(j = 0; j <= *nSupport - 1; j++) {
        for(i = 0; i <= *ntree - 1; i++) {
          cdfValuesToTrim[i] = treeCDFs[i * (*nSupport + 1) + j + 1];
        }
        if(*trimIsExterior) {
          trimmedSum = 0; 
          for(k = lo; k < hi; k++) trimmedSum += cdfValuesToTrim[index2[k]];
          trimmedEnsembleCDFs[(j + 1) * (*ntest) +  t] = trimmedSum / (double)(nTrim);
        } else {
            trimmedSum = 0;
            for(k = 0; k < lo; k++) trimmedSum += cdfValuesToTrim[index2[k]];
            for(k = hi - 1; k < *ntree; k++) trimmedSum += cdfValuesToTrim[index2[k]];
            trimmedEnsembleCDFs[(j + 1) * (*ntest) +  t] = trimmedSum / (double)(nTrim);
          }
        trimmedEnsemblePMFs[j * (*ntest) +  t] = trimmedEnsembleCDFs[(j + 1) * (*ntest) +  t] - trimmedEnsembleCDFs[j * (*ntest) +  t];
       }  
    }
    
    /* This loop finds the trimmed ensemble's mean and variance.  */
     for(j = 0; j <= *nSupport - 1; j++) {
      trimmedEnsembleMeans[t] += trimmedEnsemblePMFs[j * (*ntest) +  t] * forestSupport[j];
      trimmedEnsembleVars[t] += trimmedEnsemblePMFs[j * (*ntest) +  t] * forestSupport[j] * forestSupport[j];
      }
      trimmedEnsembleVars[t] -= trimmedEnsembleMeans[t] * trimmedEnsembleMeans[t];
    
        
    /* This statement finds the trimmed ensemble's PIT. */
    if(ytest[t] < forestSupport[0]) {
      trimmedEnsemblePITs[t] = 0;
    } else {
      indexPIT = *nSupport - 1;
      while(ytest[t] < forestSupport[indexPIT])  indexPIT -= 1;
      trimmedEnsemblePITs[t] = trimmedEnsembleCDFs[(indexPIT + 1) * (*ntest) +  t];
    }
      
    /* This loop finds the trimmed ensemble's quantiles. */ 
    zeroIntMA(index, *nQuantiles);
    zeroDoubleMA(trimmedEnsembleQuantiles, *nQuantiles);
    for(k = *nQuantiles - 1; k >= 0; k--) {
      if(k == *nQuantiles - 1) index[k] = *nSupport - 1;
      else index[k] = index[k + 1];
      /* while condition is u <= cdf value */
      while((uQuantiles[k] < trimmedEnsembleCDFs[index[k] * (*ntest) + t]) || 
            (fabs(uQuantiles[k] - trimmedEnsembleCDFs[index[k] * (*ntest) + t]) < *tol)
           ) index[k] -= 1;
      trimmedEnsembleQuantiles[k] = forestSupport[index[k]];
      index[k]++;
    } 
  
    /* These loops find the trimmed ensemble's scores: LinQuanS, LogQuanS, RPS and TMS. */
    zeroDoubleMA(trimmedEnsembleComponentScores, *nQuantiles*2);
    for(k = 0; k <= *nQuantiles - 1; k++) {
      if(trimmedEnsembleQuantiles[k] <= ytest[t]) 
        trimmedEnsembleComponentScores[k] = - uQuantiles[k] * (ytest[t] - trimmedEnsembleQuantiles[k]);
      else 
        trimmedEnsembleComponentScores[k] = - (1 - uQuantiles[k]) * (trimmedEnsembleQuantiles[k] - ytest[t]);
      trimmedEnsembleScores[t] += trimmedEnsembleComponentScores[k];   
      }
        
    for(k = 0; k <= *nQuantiles - 1; k++) {
      if(trimmedEnsembleQuantiles[k] <= ytest[t]) 
        trimmedEnsembleComponentScores[*nQuantiles + k] = - uQuantiles[k] * (log(ytest[t]) - log(trimmedEnsembleQuantiles[k]));
      else 
        trimmedEnsembleComponentScores[*nQuantiles + k] = - (1 - uQuantiles[k]) * (log(trimmedEnsembleQuantiles[k]) - log(ytest[t]));
      trimmedEnsembleScores[(*ntest) + t] += trimmedEnsembleComponentScores[*nQuantiles + k];   
      }
    
    for(j = 0; j <= *nSupport - 2; j++) {
      if(forestSupport[j] < ytest[t])
        trimmedEnsembleScores[2 * (*ntest) + t] -= trimmedEnsembleCDFs[(j + 1) * (*ntest) +  t] * trimmedEnsembleCDFs[(j + 1) * (*ntest) +  t];
      else 
        trimmedEnsembleScores[2 * (*ntest) + t] -= (1 - trimmedEnsembleCDFs[(j + 1) * (*ntest) +  t]) * (1 - trimmedEnsembleCDFs[(j + 1) * (*ntest) +  t]);
      }  
      
      trimmedEnsembleScores[3 * (*ntest) + t] = log(1/trimmedEnsembleVars[t]) - (1/trimmedEnsembleVars[t]) * (ytest[t]-trimmedEnsembleMeans[t]) * (ytest[t]-trimmedEnsembleMeans[t]);
          

  }  /* End of loop over test rows. */


} /* End of function. */

