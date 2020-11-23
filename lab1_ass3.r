require(ggplot2)
require(matlib)
require(glmnet)

set.seed(12345)

# Divide data randomly into train and test
tecator = read.csv("tecator.csv", header=TRUE)
x <- tecator[,-c(1, 103, 104)] # remove sample, protein and moisture
n = dim(x)[1]
id = sample(1:n, floor(n*0.5))

train = x[id,]
test = x[-id,]

# channel1 is column 1 in x, and channel101 in x is Fat

# ------------------------------------------------------------------------------

## Task 1
# Assume that Fat can be modeled as a linear regression in which absorbance
# characteristics (Channels) are used as features. Report the underlying
# probabilistic model, fit the linear regression to the training data and estimate
# the training and test errors. Comment on the quality of fit and prediction and
# therefore on the quality of model.

## X(X^tX)^-1X^t

M1 = lm(Fat~., data = train)
M1_fit = predict.lm(M1, newdata = train)
M2_fit = predict.lm(M1, newdata = test)

MSE = function(real, pred) {
  sqErr = (real - pred)^2
  sum_sqErr = sum(sqErr)
  N = length(sqErr)
  return(sum_sqErr/N)
}

mse_train = MSE(train$Fat, M1_fit)
mse_test = MSE(test$Fat, M2_fit)

# ------------------------------------------------------------------------------

## Task 2
# Assume now that Fat can be modeled as a LASSO regression in which all
# Channels are used as features. Report the objective function that should be
# optimized in this scenario.

## min{-loglikelihood + lambda * ||w||}
## loglikelihood = log[prod p(Y|X, w)]
## min{-sum log[p(Y|X, w)] + lambda * ||w||}
## ||w|| linear
## Y, X ??

# ------------------------------------------------------------------------------

## Task 3
# Fit the LASSO regression model to the training data. Present a plot
# illustrating how the regression coefficients depend on the log of penalty
# factor log(lambda) and interpret this plot. What value of the penalty factor can be
# chosen if we want to select a model with only three features?

obs = train[,1:100]
resp = train$Fat
predLasso = glmnet(as.matrix(obs), resp, alpha = 1, family = "gaussian")
plot(predLasso, xvar = "lambda", label = TRUE,
     main = "LASSO regression")
print(paste("Lambda for three features:",
            predLasso$lambda[22], predLasso$lambda[23], predLasso$lambda[24]))

# ------------------------------------------------------------------------------

## Task 4
# Present a plot of how degrees of freedom depend on the penalty parameter.
# Is the observed trend expected?

plot(predLasso$lambda, predLasso$df, type = "l", col = "red", 
     xlab = "lambda", ylab = "degrees of freedom",
     main = "Degrees of Freedom vs Penalty Factor")

# ------------------------------------------------------------------------------

## Task 5
# Repeat step 3 but fit Ridge instead of the LASSO regression and compare the
# plots from steps 3 and 5. Conclusions?

predRidge = glmnet(as.matrix(obs), resp, alpha = 0, family = "gaussian") 
plot(predRidge, xvar = "lambda", label = TRUE,
     main = "Ridge regression")

# ------------------------------------------------------------------------------

## Task 6
# Use cross-validation to compute the optimal LASSO model. Present a plot
# showing the dependence of the CV score on log(lambda) and comment how the CV
# score changes with log(lambda). Report the optimal lambda and how many variables were
# chosen in this model. Comment whether the selected lambda value is statistically
# significantly better than log(lambda) = -2. Finally, create a scatter plot of the
# original test versus predicted test values for the model corresponding to optimal
# lambda and comment whether the model predictions are good.

cv_model_lasso = cv.glmnet(as.matrix(obs), resp, family = "gaussian", alpha = 1)
print(paste("Best lambda:", cv_model_lasso$lambda.min,
            ", no. of variables:", length(cv_model_lasso$nzero)))
plot(cv_model_lasso, main = "Cross-validation for LASSO")
coef(cv_model_lasso, s = "lambda.min")

pred = predict(cv_model_lasso, newx = as.matrix(test[,1:100]), s ="lambda.min")
plot(test$Fat, col = "blue", xlab = "Sample", ylab = "Value of fat",
     main = "Original test vs Predicted test")
points(pred, col = "red")

# ------------------------------------------------------------------------------

## Task 7
# Use the feature values from test data (the portion of test data with Channel
# columns) and the optimal LASSO model from step 6 to generate new target
# values. (Hint: use rnorm() and compute sigma as standard deviation of residuals
# from train data predictions). Make a scatter plot of original Fat in test data
# versus newly generated ones. Comment on the quality of the data generation.

predTest = predict(cv_model_lasso, newx = as.matrix(train[,1:100]), s ="lambda.min")
res = predTest - train$Fat
sigma = sd(res)
normDist = rnorm(pred, sd = sigma)
newGen = normDist + pred
plot(test$Fat, col = "blue", xlab = "Sample", ylab = "Value of fat",
     main = "Original test vs Generated test")
points(newGen, col = "red")