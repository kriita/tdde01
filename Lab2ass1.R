require(ggplot2)
require(MASS)
require(matlib)
require(glmnet)
require(mvtnorm)
require(nnet)

set.seed(12345)

## Task 1
# Make a scatterplot of Sepal Width versus Sepal Length where observations are
# colored by Species. Do you think that this data is easy to classify by linear
# discriminant analysis? Motivate your answer

# take out only interesting data
# x = (length width)
sepal = iris[,-c(3, 4, 5)]
setosa = sepal[1:50,]
versicolor = sepal[51:100,]
virginica = sepal[101:150,]

plot(setosa[, 1], setosa[, 2], col = "blue", pch = 21,
     xlim = c(4.3, 7.9), ylim = c(1.8, 4.7), bg = "blue",
     main = "Sepal length vs Sepal width", xlab = "Length", ylab = "Width")
points(versicolor[, 1], versicolor[, 2], col = "red", pch = 21, bg = "red")
points(virginica[, 1], virginica[, 2], col = "green", pch = 21, bg = "green")
legend("topleft", title = "Original",
       legend = c("Setosa", "Versicolor", "Virginica"),
       col = c("blue", "red", "green"), lty = 1, lwd = 3)

# ------------------------------------------------------------------------------

## Task 2
# Use basic R functions only to implement Linear Discriminant Analysis between
# the three species based on variables Sepal Length and Sepal Width:

# a) Compute mean, covariance matrices (use cov()) and prior probabilities
# per class and report them
# b) Compute overall (pooled) covariance matrix and report it
# c) Report the probabilistic model for the LDA
# d) Compute discriminant functions for each class
# e) Compute equations of decision boundaries between classes and report
# them

# Do estimated covariance matrices seem to fulfill LDA assumptions?

# a)
# mean_k = (length, width)
mean_tot = as.matrix(c(mean(sepal[, 1]), mean(sepal[, 2])))
mean_setosa = as.matrix(c(mean(setosa[, 1]), mean(setosa[, 2])))
mean_versicolor = as.matrix(c(mean(versicolor[, 1]), mean(versicolor[, 2])))
mean_virginica = as.matrix(c(mean(virginica[, 1]), mean(virginica[, 2])))

priorProb = 50/150 # 50 of each species, three different species and 150 samples

cov_tot = cov(sepal)
cov_setosa = cov(setosa)
cov_versicolor = cov(versicolor)
cov_virginica = cov(virginica)

# b)
pooledCov = priorProb * (cov_setosa + cov_versicolor + cov_virginica)

# d)
# delta_k = x^T * w_k + w_0k
w_tot = inv(cov_tot) %*% mean_tot
w_setosa = inv(cov_setosa) %*% mean_setosa
w_versicolor = inv(cov_versicolor) %*% mean_versicolor
w_virginica = inv(cov_virginica) %*% mean_virginica

w0_tot = -1/2 * t(mean_tot) %*% w_tot + log(1)
w0_setosa = -1/2 * t(mean_setosa) %*% w_setosa + log(priorProb)
w0_versicolor = -1/2 * t(mean_versicolor) %*% w_versicolor + log(priorProb)
w0_virginica = -1/2 * t(mean_virginica) %*% w_virginica + log(priorProb)

discriminant = function(data, w, w0){
  as.matrix(data) %*% w - c(w0)
}

discr_tot = max(discriminant(sepal, w_tot, w0_tot))
discr_setosa = max(discriminant(setosa, w_setosa, w0_setosa))
discr_versicolor = max(discriminant(versicolor, w_versicolor, w0_versicolor))
discr_virginica = max(discriminant(virginica, w_virginica, w0_virginica))

# ------------------------------------------------------------------------------

## Task 3
# Use discriminant functions from step 2 to predict the species from the original
# data and make a scatterplot of Sepal Length versus Sepal Width in which color
# shows the predicted Species. Estimate the misclassification rate of the prediction.
# Comment on the quality of classification.
# Afterwards, perform the LDA analysis with lda() function and investigate
# whether you obtain the same test error by using this package.
# Should it be same?

#pred_setosa = 
#pred_versicolor = 
#pred_versicolor = 
  
plot(pred_setosa[, 1], pred_setosa[, 2], col = "blue", pch = 21,
     xlim = c(4.3, 7.9), ylim = c(1.7, 5), bg = "blue",
     main = "Sepal length vs Sepal width", xlab = "Length", ylab = "Width")
points(pred_versicolor[, 1], pred_versicolor[, 2],
       col = "red", pch = 21, bg = "red")
points(pred_virginica[, 1], pred_virginica[, 2],
       col = "green", pch = 21, bg = "green")
legend("topleft", title = "Predicted",
       legend = c("Setosa", "Versicolor", "Virginica"),
       col = c("blue", "red", "green"), lty = 1, lwd = 3)

#misclass_lda = 

LDA = lda(Species ~ Sepal.Length + Sepal.Width, data = iris)
print(LDA)
plot(LDA)

# ------------------------------------------------------------------------------

## Task 4
# Use Models reported in 2c to generate new data of this kind with the same total
# number of cases as in the original data (hint: use sample() and rmvnorm() from
# package mvtnorm). Make a scatterplot of the same kind as in step 1 but for the
# new data and compare it with the plots for the original and the predicted data.
# Conclusions?

new_setosa = rmvnorm(n = 50, mean = mean_setosa, sigma = cov_setosa)
new_versicolor = rmvnorm(n = 50, mean = mean_versicolor, sigma = cov_versicolor)
new_virginica = rmvnorm(n = 50, mean = mean_virginica, sigma = cov_virginica)

plot(new_setosa[, 1], new_setosa[, 2], col = "blue", pch = 21,
     xlim = c(4.3, 7.9), ylim = c(1.7, 5), bg = "blue",
     main = "Sepal length vs Sepal width", xlab = "Length", ylab = "Width")
points(new_versicolor[, 1], new_versicolor[, 2], col = "red", pch = 21, bg = "red")
points(new_virginica[, 1], new_virginica[, 2], col = "green", pch = 21, bg = "green")
legend("topleft", title = "Generated",
       legend = c("Setosa", "Versicolor", "Virginica"),
       col = c("blue", "red", "green"), lty = 1, lwd = 3)

# ------------------------------------------------------------------------------

## Task 5
# Make a similar kind of classification by logistic regression (use function
# multinom() from nnet package), plot the classified data and compute the
# misclassification error. Compare these results with the LDA results.

log_reg = multinom(Species ~ Sepal.Length + Sepal.Width, data = iris)
#plot()
#misclass_log_reg = 
