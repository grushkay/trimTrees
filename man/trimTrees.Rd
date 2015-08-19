\name{trimTrees}

\alias{trimTrees}

\title{Trimmed Opinion Pools of Trees in Random Forest}

\description{This function creates point and probability forecasts from the trees in a random forest using Jose et al.'s trimmed opinion pool, a trimmed average of the trees' empirical cumulative distribution functions (cdf).  For tuning purposes, the user can input the trimming level used in this trimmed average and then compare the scores of the trimmed and untrimmed opinion pools, or ensembles.}

\usage{
trimTrees(xtrain, ytrain, xtest, ytest=NULL, ntree = 500, 
          mtry = if (!is.null(ytrain) && !is.factor(ytrain)) 
          max(floor(ncol(xtrain)/3), 1) else floor(sqrt(ncol(xtrain))), 
          nodesize = if (!is.null(ytrain) && !is.factor(ytrain)) 5 else 1, 
          trim = 0,trimIsExterior = TRUE, 
          uQuantiles = seq(0.05, 0.95, 0.05), methodIsCDF = TRUE)
}

\arguments{
  \item{xtrain}{
A data frame or a matrix of predictors for the training set.
}
  \item{ytrain}{
A response vector for the training set.  If a factor, classification is assumed, otherwise regression is assumed. 
}
  \item{xtest}{
A data frame or a matrix of predictors for the testing set.
}
  \item{ytest}{
A response vector for the testing set. If no testing set is passed, probability integral transform (PIT) values and scores will be returned as \code{NA}s. 
}
  \item{ntree}{
Number of trees to grow.
}
  \item{mtry}{
Number of variables randomly sampled as candidates at each split.
}
  \item{nodesize}{
Minimum size of terminal nodes.
}
  \item{trim}{
The trimming level used in the trimmed average of the trees' empirical cdfs.  For the cdf approach, the trimming level is the fraction of cdfs values to be trimmed from each end of the ordered vector of cdf values (for each support point) before the average is computed.  For the moment approach, the trees' means are computed, ordered, and trimmed.  The trimmed opinion pool using the moment approach is an average of the remaining trees. 
}
  \item{trimIsExterior}{
If \code{TRUE}, the trimming is done exteriorly, or from the ends of the ordered vector.  If \code{FALSE}, the trimming is done interiorly, or from the middle of the ordered vector.
}
  \item{uQuantiles}{
A vector of probabilities in a strictly increasing order and between 0 and 1.  For instance, if \code{uQuantiles=c(0.25,0.75)}, then the 0.25-quantile and the 0.75-quantile of the trimmed and untrimmed ensembles are scored.
}
  \item{methodIsCDF}{
If \code{TRUE}, the method for forming the trimmed opinion pool is according to the cdf approach in Jose et al (2014).  If \code{FALSE}, the moment approach is used.  
}
}

\value{
An object of class \code{trimTrees}, which is a list with the following components:
\item{forestSupport}{Possible points of support for the trees and ensembles.}
\item{treeValues}{For the last testing set row, this component outputs each tree's \code{ytrain} values (not necessarily unique) that are both inbag and in the \code{xtest}'s terminal node.  Note that the \code{ytrain} values may not be unique.  This component is an \code{ntrain}-by-\code{ntree} matrix where \code{ntrain} is the number of rows in the training set.}
\item{treeCounts}{For the last testing set row, each tree's counts of \code{treeValues} and lists them by their unique values.  This component is an \code{nSupport}-by-\code{ntree} matrix.  \code{nSupport} is the number of unique \code{ytrain} values, or support points of the forest.}
\item{treeCumCounts}{Cumulative tally of \code{treeCounts} of dimension \code{nSupport+1}-by-\code{ntree}.}
\item{treeCDFs}{Each tree's empirical cdf based on \code{treeCumCounts} for the last testing set row only.  This component is an \code{nSupport+1}-by-\code{ntree} matrix.  Note that the first row in this matrix is all zeros.}
\item{treePMFs}{Each tree's empirical probability mass function (pmf) for the last testing set row.  This component is an \code{nSupport}-by-\code{ntree} matrix.}
\item{treeMeans}{For each testing set row, each tree's mean according to its empirical pmf.  This component is an \code{ntest}-by-\code{ntree} matrix where \code{ntest} is the number of rows in the testing set.}
\item{treeVars}{For each testing set row, each tree's variance according to its empirical pmf.  This component is an \code{ntest}-by-\code{ntree} matrix.}
\item{treePITs}{For each testing set row, each tree's probability integral transform (PIT), the empirical cdf evaluated at the realized \code{ytest} value.  This component is an \code{ntest}-by-\code{ntree} matrix. If \code{ytest} is \code{NULL}, \code{NA}s are returned.}
\item{treeQuantiles}{For the last testing set row, each tree's quantiles -- one for each element in \code{uQuantiles}, the empirical cdf evaluated at the realized \code{ytest} value.  This component is an \code{ntree}-by-\code{nQuantile} matrix where \code{nQuantile} is the number of elements in \code{uQuantiles}.}
\item{treeFirstPMFValues}{For each testing set row, this component outputs the pmf value on the minimum (or first) support point in the forest.  For binary classification, this corresponds to the probability that the minimum (or first) support point will occur.  This component's dimension is \code{ntest}-by-\code{ntree}.  It is useful for generating calibration curves (stated probabilities in bins vs. their observed frequencies) for binary classification.}
\item{bracketingRate}{For each testing set row, the bracketing rate from Larrick et al. (2012) is computed as \code{2*p*(1-p)} where \code{p} is the fraction of trees' means above the \code{ytest} value. If \code{ytest} is \code{NULL}, \code{NA}s are returned.}
\item{bracketingRateAllPairs}{The average bracketing rate across all testing set rows for each pair of trees. This component is a symmetric \code{ntree}-by-\code{ntree} matrix. If \code{ytest} is \code{NULL}, \code{NA}s are returned.}
\item{trimmedEnsembleCDFs}{For each testing set row, the trimmed ensemble's forecast of \code{ytest} in the form of a cdf. This component is an \code{ntest}-by-\code{nSupport + 1} matrix. \code{nSupport} is the number of unique \code{ytrain} values, or support points of the forest.}
\item{trimmedEnsemblePMFs}{For each testing set row, the trimmed ensemble's pmf. This component is an \code{ntest}-by-\code{nSupport} matrix.}
\item{trimmedEnsembleMeans}{For each testing set row, the trimmed ensemble's mean. This component is an \code{ntest} vector.}
\item{trimmedEnsembleVars}{For each testing set row, the trimmed ensemble's variance.}
\item{trimmedEnsemblePITs}{For each testing set row, the trimmed ensemble's probability integral transform (PIT), the empirical cdf evaluated at the realized \code{ytest} value. If \code{ytest} is \code{NULL}, \code{NA}s are returned.}
\item{trimmedEnsembleQuantiles}{For the last testing set row, the trimmed ensemble's quantiles -- one for each element in \code{uQuantiles}.}
\item{trimmedEnsembleComponentScores}{For the last testing set row, the components of the trimmed ensemble's linear and log quantile scores.If \code{ytest} is \code{NULL}, \code{NA}s are returned.}
\item{trimmedEnsembleScores}{For each testing set row, the trimmed ensemble's linear and log quantile scores, ranked probability score, and two-moment score.  See Jose and Winkler (2009) for a description of the linear and log quantile scores.  See Gneiting and Raftery (2007) for a description of the ranked probability score.  The two-moment score is the score in Equation 27 of Gneiting and Raftery (2007). If \code{ytest} is \code{NULL}, \code{NA}s are returned.}
\item{untrimmedEnsembleCDFs}{For each testing set row, the linear opinion pool's, or untrimmed ensemble's, forecast of \code{ytest} in the form of a cdf.}
\item{untrimmedEnsemblePMFs}{For each testing set row, the untrimmed ensemble's pmf.}
\item{untrimmedEnsembleMeans}{For each testing set row, the untrimmed ensemble's mean.}
\item{untrimmedEnsembleVars}{For each testing set row, the untrimmed ensemble's variance.}
\item{untrimmedEnsemblePITs}{For each testing set row, the untrimmed ensemble's probability integral transform (PIT), the empirical cdf evaluated at the realized \code{ytest} value. If \code{ytest} is \code{NULL}, \code{NA}s are returned.}
\item{untrimmedEnsembleQuantiles}{For the last testing set row, the untrimmed ensemble's quantiles -- one for each element in \code{uQuantiles}.}
\item{untrimmedEnsembleComponentScores}{For the last testing set row, the components of the untrimmed ensemble's linear and log quantile scores. If \code{ytest} is \code{NULL}, \code{NA}s are returned.}
\item{untrimmedEnsembleScores}{For each testing set row, the untrimmed ensemble's linear and log quantile scores, ranked probability score, and two-moment score. If \code{ytest} is \code{NULL}, \code{NA}s are returned.}
}

\references{
Gneiting T, Raftery AE. (2007). Strictly proper scoring rules, prediction, and estimation. Journal of the American Statistical Association 102 359-378.

Jose VRR, Grushka-Cockayne Y, Lichtendahl KC Jr. (2014). Trimmed opinion pools and the crowd's calibration problem. Management Science 60 463-475.

Jose VRR, Winkler RL (2009). Evaluating quantile assessments. Operations Research 57 1287-1297.

Grushka-Cockayne Y, Jose VRR, Lichtendahl KC Jr. (2014). Ensembles of overfit and overconfident forecasts, working paper.

Larrick RP, Mannes AE, Soll JB (2011). The social psychology of the wisdom of crowds.  In J.I. Krueger, ed., Frontiers in Social Psychology: Social Judgment and Decision Making. New York: Psychology Press, 227-242.
}
\author{
Yael Grushka-Cockayne, Victor Richmond R. Jose, Kenneth C. Lichtendahl Jr., and Huanghui Zeng.
}

\seealso{
\code{\link{hitRate}}, \code{\link{cinbag}}
}
\examples{
# Load the data
set.seed(201) # Can be removed; useful for replication
data <- as.data.frame(mlbench.friedman1(500, sd=1))
summary(data)

# Prepare data for trimming
train <- data[1:400, ]
test <- data[401:500, ]
xtrain <- train[,-11]  
ytrain <- train[,11]
xtest <- test[,-11]
ytest <- test[,11]
      
# Option 1. Run trimTrees with responses in testing set.
set.seed(201) # Can be removed; useful for replication
tt1 <- trimTrees(xtrain, ytrain, xtest, ytest, trim=0.15)

#Some outputs from trimTrees: scores, hit rates, PIT densities.
colMeans(tt1$trimmedEnsembleScores)
colMeans(tt1$untrimmedEnsembleScores)
mean(hitRate(tt1$treePITs))
hitRate(tt1$trimmedEnsemblePITs)
hitRate(tt1$untrimmedEnsemblePITs)
hist(tt1$trimmedEnsemblePITs, prob=TRUE)
hist(tt1$untrimmedEnsemblePITs, prob=TRUE)

# Option 2. Run trimTrees without responses in testing set. 
# In this case, scores, PITs, or hit rates will not be available.
set.seed(201) # Can be removed; useful for replication
tt2 <- trimTrees(xtrain, ytrain, xtest, trim=0.15)

# Some outputs from trimTrees: cdfs for last test value.
plot(tt2$trimmedEnsembleCDFs[100,],type="l",col="red",ylab="cdf",xlab="y") 
lines(tt2$untrimmedEnsembleCDFs[100,])
legend(275,0.2,c("trimmed", "untrimmed"),col=c("red","black"),lty = c(1, 1))
title("CDFs of Trimmed and Untrimmed Ensembles")

# Compare the CDF and moment approaches to trimming the trees.
ttCDF <- trimTrees(xtrain, ytrain, xtest, trim=0.15, methodIsCDF=TRUE)
ttMA <- trimTrees(xtrain, ytrain, xtest, trim=0.15, methodIsCDF=FALSE)
plot(ttCDF$trimmedEnsembleCDFs[100,], type="l", col="red", ylab="cdf", xlab="y") 
lines(ttMA$trimmedEnsembleCDFs[100,])
legend(275,0.2,c("CDF Approach", "Moment Approach"), col=c("red","black"),lty = c(1, 1))
title("CDFs of Trimmed Ensembles")
}

% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{classif}% at least one, from doc/KEYWORDS
\keyword{randomForest}
\keyword{regression}
\keyword{tree}
% __ONLY ONE__ keyword per line
