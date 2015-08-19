\name{cinbag}

\alias{cinbag}

\title{Modified Classification and Regression with Random Forest
}

\description{\code{cinbag} implements a modified random forest algorithm (based on the source code from the randomForest package by Andy Liaw and Matthew Wiener and on the original Fortran code by Leo Breiman and Adele Cutler) to return the number of times a row appears in a tree's bag.  \code{cinbag} returns a \code{randomForest} object, e.g., \code{rfobj}, with an additional output, a matrix with inbag counts (rows) for each tree (columns).  For instance, \code{rfobj$inbagCount} is similar to \code{rfobj$inbag}, but with inbag counts instead of inbag indicators.}

\usage{
      cinbag(x, y=NULL,  xtest=NULL, ytest=NULL, ntree=500,
             mtry=if (!is.null(y) && !is.factor(y))
             max(floor(ncol(x)/3), 1) else floor(sqrt(ncol(x))),
             replace=TRUE, classwt=NULL, cutoff, strata,
             sampsize = if (replace) nrow(x) else ceiling(.632*nrow(x)),
             nodesize = if (!is.null(y) && !is.factor(y)) 5 else 1,
             maxnodes = NULL,
             importance=FALSE, localImp=FALSE, nPerm=1,
             proximity, oob.prox=proximity,
             norm.votes=TRUE, do.trace=FALSE,
             keep.forest=!is.null(y) && is.null(xtest), corr.bias=FALSE,
             keep.inbag=FALSE, ...)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{x}{a data frame or a matrix of predictors, or a formula
    describing the model to be fitted (for the
    \code{print} method, an \code{randomForest} object).}
  \item{y}{A response vector.  If a factor, classification is assumed,
    otherwise regression is assumed.  If omitted, \code{randomForest}
    will run in unsupervised mode.}
  \item{xtest}{a data frame or matrix (like \code{x}) containing
    predictors for the test set.}
  \item{ytest}{response for the test set.}
  \item{ntree}{Number of trees to grow.  This should not be set to too
    small a number, to ensure that every input row gets predicted at
    least a few times.}
  \item{mtry}{Number of variables randomly sampled as candidates at each
    split.  Note that the default values are different for
    classification (sqrt(p) where p is number of variables in \code{x})
    and regression (p/3).}
  \item{replace}{Should sampling of cases be done with or without
    replacement?}
  \item{classwt}{Priors of the classes.  Need not add up to one.
    Ignored for regression.}
  \item{cutoff}{(Classification only)  A vector of length equal to
    number of classes.  The `winning' class for an observation is the
    one with the maximum ratio of proportion of votes to cutoff.
    Default is 1/k where k is the number of classes (i.e., majority vote
    wins).}
  \item{strata}{A (factor) variable that is used for stratified sampling.}
  \item{sampsize}{Size(s) of sample to draw.  For classification, if
    sampsize is a vector of the length the number of strata, then
    sampling is stratified by strata, and the elements of sampsize
    indicate the numbers to be drawn from the strata.}
  \item{nodesize}{Minimum size of terminal nodes.  Setting this number
    larger causes smaller trees to be grown (and thus take less time).
    Note that the default values are different for classification (1)
    and regression (5).}
  \item{maxnodes}{Maximum number of terminal nodes trees in the forest
  can have.  If not given, trees are grown to the maximum possible
	(subject to limits by \code{nodesize}).  If set larger than maximum
	possible, a warning is issued.}
  \item{importance}{Should importance of predictors be assessed? }
  \item{localImp}{Should casewise importance measure be computed?
    (Setting this to \code{TRUE} will override \code{importance}.) }
  \item{nPerm}{Number of times the OOB data are permuted per tree for
    assessing variable importance.  Number larger than 1 gives slightly
    more stable estimate, but not very effective.  Currently only
    implemented for regression.}
  \item{proximity}{Should proximity measure among the rows be
    calculated?}
  \item{oob.prox}{Should proximity be calculated only on ``out-of-bag''
    data?}
  \item{norm.votes}{If \code{TRUE} (default), the final result of votes
    are expressed as fractions.  If \code{FALSE}, raw vote counts are
    returned (useful for combining results from different runs).
    Ignored for regression.}
  \item{do.trace}{If set to \code{TRUE}, give a more verbose output as
    \code{randomForest} is run.  If set to some integer, then running
    output is printed for every \code{do.trace} trees.}
  \item{keep.forest}{If set to \code{FALSE}, the forest will not be
    retained in the output object.  If \code{xtest} is given, defaults
    to \code{FALSE}.}
  \item{corr.bias}{perform bias correction for regression?  Note:
    Experimental.  Use at your own risk.}
  \item{keep.inbag}{Should an \code{n} by \code{ntree} matrix be
    returned that keeps track of which samples are ``in-bag'' in which
    trees (but not how many times, if sampling with replacement)}
  \item{...}{optional parameters to be passed to the low level function
    \code{cinbag.default}.}
}

\value{An object of class \code{randomForest}, which is a list with the
  following components:
  \item{call}{the original call to \code{randomForest}}
  \item{type}{one of \code{regression}, \code{classification}, or
    \code{unsupervised}.}
  \item{predicted}{the predicted values of the input data based on
    out-of-bag samples.}
  \item{importance}{a matrix with \code{nclass} + 2 (for classification)
    or two (for regression) columns.  For classification, the first
    \code{nclass} columns are the class-specific measures computed as
    mean descrease in accuracy.  The \code{nclass} + 1st column is the
    mean descrease in accuracy over all classes.  The last column is the
    mean decrease in Gini index.  For Regression, the first column is
    the mean decrease in accuracy and the second the mean decrease in MSE.
    If \code{importance=FALSE}, the last measure is still returned as a
    vector.}
  \item{importanceSD}{The ``standard errors'' of the permutation-based
    importance measure.  For classification, a \code{p} by \code{nclass
      + 1} matrix corresponding to the first \code{nclass + 1} columns
    of the importance matrix.  For regression, a length \code{p} vector.}
  \item{localImp}{a p by n matrix containing the casewise importance
    measures, the [i,j] element of which is the importance of i-th
    variable on the j-th case. \code{NULL} if \code{localImp=FALSE}.}
  \item{ntree}{number of trees grown.}
  \item{mtry}{number of predictors sampled for spliting at each node.}
  \item{forest}{(a list that contains the entire forest; \code{NULL} if
    \code{randomForest} is run in unsupervised mode or if
    \code{keep.forest=FALSE}.}
  \item{err.rate}{(classification only) vector error rates of the
    prediction on the input data, the i-th element being the (OOB) error rate
    for all trees up to the i-th.}
  \item{confusion}{(classification only) the confusion matrix of the
    prediction (based on OOB data).}
  \item{votes}{(classification only) a matrix with one row for each
    input data point and one column for each class, giving the fraction
    or number of (OOB) `votes' from the random forest.}
  \item{oob.times}{number of times cases are `out-of-bag' (and thus used
    in computing OOB error estimate)}
  \item{proximity}{if \code{proximity=TRUE} when
    \code{randomForest} is called, a matrix of proximity measures among
    the input (based on the frequency that pairs of data points are in
    the same terminal nodes).}
  \item{mse}{(regression only) vector of mean square errors: sum of squared
    residuals divided by \code{n}.}
  \item{rsq}{(regression only) ``pseudo R-squared'': 1 - \code{mse} /
    Var(y).}
  \item{test}{if test set is given (through the \code{xtest} or additionally
  \code{ytest} arguments), this component is a list which contains the
  corresponding \code{predicted}, \code{err.rate}, \code{confusion},
  \code{votes} (for classification) or \code{predicted}, \code{mse} and
  \code{rsq} (for regression) for the test set.  If
  \code{proximity=TRUE}, there is also a component, \code{proximity},
  which contains the proximity among the test set as well as proximity
  between test and training data.}
  \item{inbag}{An indicator (1 or 0) for each training set row and each tree.  The indicator is 1 if the training set row is in the tree's bag and is 0 otherwise. Note that this value is not listed in the original \code{randomForest} function's output, although it is implemented.}
  \item{inbagCount}{A count for each training set row and each tree.  The count is the number of times the training set row is in the tree's bag.  This output is not available in the original \code{randomForest} package.  The purpose of the \code{cinbag} function is to augment the \code{randomForest} function so that it returns inbag counts.  These counts are necessary for computing and ensembling the trees' empirical cumulative distribution functions.}
}


\note{\code{cinbag}'s source files call the C functions \code{classRFmod.c} and \code{regRFmod.c}, which are slightly modified versions of the \code{randomForest}'s source files \code{classRF.c} and \code{regRF.c}, respectively.}

\references{
  Breiman L (2001). Random forests. Machine Learning 45 5-32.

  Breiman L (2002). Manual on setting up, using, and understanding random forests V3.1. \url{http://oz.berkeley.edu/users/breiman/Using_random_forests_V3.1.pdf}.
}

\author{Yael Grushka-Cockayne, Victor Richmond R. Jose, Kenneth C. Lichtendahl Jr. and Huanghui Zeng, based on the source code from the randomForest package by Andy Liaw and Matthew Wiener and on the original Fortran code by Leo Breiman and Adele Cutler.}

\seealso{\code{\link{trimTrees}}, \code{\link{hitRate}}}
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
      
# Run cinbag
set.seed(201) # Can be removed; useful for replication
rf <- cinbag(xtrain, ytrain, ntree=500, nodesize=5, mtry=3, keep.inbag=TRUE)
rf$inbag[,1] # First tree's inbag indicators 
rf$inbagCount[,1] # First tree's inbag counts
}

% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{classif}% at least one, from doc/KEYWORDS
\keyword{randomForest}
\keyword{regression}
\keyword{tree}
