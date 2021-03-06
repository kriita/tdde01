## Import data ##
library(glmnet)
setwd("~/Dropbox/Skola/LiU/TDDE01/tdde01")
data = read.csv("./upg1_data/parkinsons.csv")

# ------------------------------------------------------------------------------

# TASK 2 - Divide data into training and test data (60/40) #
data = scale(data)
n = dim(data)[1]
set.seed(12345)
id = sample(1:n, floor(n*0.6))
train = data[id,]
train_motor = train[,5]
train = train[,-5]
test = data[-id,]
test_motor = test[,5]
test = test[,-5]

# ------------------------------------------------------------------------------

## TASK 3 - Implement functions using the R std library ##
# The following functions should be implemented:
# a) loglikelihood(w,sigma)
# b) ridge(w,sigma,lambda)
# c) ridgeOpt(lambda)
# d) df(lambda)

# a)

loglikelihood <- function(w,sigma){ #https://www.statlect.com/fundamentals-of-statistics/normal-distribution-maximum-likelihood
  return(-1/(2*sigma^2)*(sum((train_motor - train%*%w)^2)) - length(train_motor)/2*log(sigma^2) - length(train_motor)/2*log(2*pi))
}

# b)
ridge <- function(w, sigma, lambda){
  return(- loglikelihood(w = w, sigma = sigma) + lambda * sum(w^2)) # add penalty to log-likelihood
}

# c)

ridgeOpt <- function(lambda){
  return(optim(par = rep(1,22), fn = function(x) { 
    w <- c(x[1:21])
    sigma <- c(x[22])
    return(ridge(w=w,sigma=sigma,lambda))
  }, method = "BFGS")) # Start values is a column vector with 1s
}

# d)
df <- function(lambda){
  train=as.matrix(train)
  return( sum(diag( train %*% (t(train)%*%train + lambda*diag(ncol(train)))^(-1) %*% t(train) ))) # df=trace(P), where P is the hat matrix
}

# ------------------------------------------------------------------------------

## TASK 4 & TASK 5 - calculate measurements for model analysis ##
# Task 4: calculate  MSE for the training and testing data for lambdas 1, 100,
# and 1000, and store these in the mse_train and mse_test vectors, respectively.
# Task 5: calculate the AIC for the model for these lambdas, and store these in
# the aic_vec vector.
#
# Discussion and presentation of these results is covered in the lab report.

# AIC def= 2df - loglik
aic = function(w, sigma, lambda){
  return(2 * df(lambda) - 2*loglikelihood(w, sigma))
}

# Initialize testing variables
lambdas <- c(1, 100, 1000)
w_vec <- rep(0,length(lambdas))
aic_vec <- rep(0,length(lambdas))
mse_train <- rep(0,length(lambdas))
mse_test <- rep(0,length(lambdas))
i = 1
mse_train = 0

# Loop through the lambas, using ridgeOpt to find the optimal w and sigma, using
# these in the calculations for the MSE of the model compared to the training
# and testing data, and then calculating the AIC of the results.
for (lambda in lambdas) {
  optlambda = ridgeOpt(lambda)
  optlambda = optlambda[[1]]
  w = optlambda[1:21]
  sigma = optlambda[22]
  mse_train[i] = sum((train%*%w - train_motor)^2)/length(train_motor)
  mse_test[i] = sum((test%*%w - test_motor)^2)/length(test_motor)
  aic_vec[i] <- aic(w, sigma, lambda)
  i = i + 1
}

