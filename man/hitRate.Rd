\name{hitRate}
\alias{hitRate}
\title{Empirical Hit Rates for a Crowd of Forecasters}
\description{This function calculates the empirical hit rates for a crowd of forecasters over a testing set.  The function takes as its arguments the forecasters' probability integral transform (PIT) values -- one for each testing set row -- and the prediction interval of interest.}
\usage{
hitRate(matrixPIT, interval = c(0.25, 0.75))
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{matrixPIT}{
  A \code{ntest}-by-\code{nForecaster} matrix of PIT values where \code{ntest} is the number of rows in the testing set and \code{nForecaster} is the number of forecasters.  Each column represents a different forecaster's PITs for the testing set.  A PIT value is the forecaster's cdf evaluated at the realization of the response in the testing set.}

\item{interval}{
Prediction interval of interest.  The default \code{interval=c(0.25, 0.75)} is the central 50\% prediction interval.}
}

\value{
\item{HR}{An \code{nForecaster} vector of empirical hit rates -- one for each forecaster.  A forecaster's empirical hit rate is the percentage of PIT values that fall within [\code{interval[1]},\code{interval[2]}], e.g., [0.25,0.75] according to the default.}
}
\references{
Grushka-Cockayne Y, Jose VRR, Lichtendahl KC Jr. (2014). Ensembles of overfit and overconfident forecasts, working paper.
}

\author{
Yael Grushka-Cockayne, Victor Richmond R. Jose, Kenneth C. Lichtendahl Jr., and Huanghui Zeng.
}

\seealso{
\code{\link{trimTrees}}, \code{\link{cinbag}}
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
      
# Run trimTrees
set.seed(201) # Can be removed; useful for replication
tt <- trimTrees(xtrain, ytrain, xtest, ytest, trim=0.15)

# Outputs from trimTrees
mean(hitRate(tt$treePITs))
hitRate(tt$trimmedEnsemblePITs)
hitRate(tt$untrimmedEnsemblePITs)
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{classif}% at least one, from doc/KEYWORDS
\keyword{randomForest}
\keyword{regression}
\keyword{tree}
