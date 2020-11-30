require(ggplot2)
require(MASS)
require(matlib)
require(glmnet)
require(mvtnorm)
require(nnet)

set.seed(12345)
attach(iris) # put the iris data in your current work directory

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
# e) Compute equations of decision boundaries between classes and report them
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

# c)
# x|y = C_i, mu_i, Sigma ~ N(mu_i, Sigma)
# y|pi ~ Multinomial(pi_1, ..., pi_k)

# p(Y = C_i | X) = exp(w_i^T * x) / sum_j=1^K exp(w_j^T * x)
# w_0i = -1/2 * mu_i^T * Sigma^-1 * mu_i + log(pi_i)
# w_i = Sigma^-1 * mu_i

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

################################################################################
# osäker

discr_tot = max(discriminant(sepal, w_tot, w0_tot))
discr_setosa = max(discriminant(setosa, w_setosa, w0_setosa))
discr_versicolor = max(discriminant(versicolor, w_versicolor, w0_versicolor))
discr_virginica = max(discriminant(virginica, w_virginica, w0_virginica))

one = max(discriminant(sepal, w_setosa, w0_setosa))
two = max(discriminant(sepal, w_versicolor, w0_versicolor))
three = max(discriminant(sepal, w_virginica, w0_virginica))

################################################################################

# e)
# w_1 * x + w_01 = w_2 * x + w_02
# w_1 * x + w_01 = w_3 * x + w_03
# w_2 * x + w_02 = w_3 * x + w_03
# x = (length, width)

# ------------------------------------------------------------------------------

## Task 3
# Use discriminant functions from step 2 to predict the species from the original
# data and make a scatterplot of Sepal Length versus Sepal Width in which color
# shows the predicted Species. Estimate the misclassification rate of the
# prediction. Comment on the quality of classification.
# Afterwards, perform the LDA analysis with lda() function and investigate
# whether you obtain the same test error by using this package.
# Should it be same?

################################################################################
# ej klar

pred = ?
  
plot(?, col = "blue", pch = 21,
     xlim = c(4.3, 7.9), ylim = c(1.7, 5), bg = "blue",
     main = "Sepal length vs Sepal width", xlab = "Length", ylab = "Width")
points(?,
       col = "red", pch = 21, bg = "red")
points(?,
       col = "green", pch = 21, bg = "green")
legend("topleft", title = "Predicted",
       legend = c("Setosa", "Versicolor", "Virginica"),
       col = c("blue", "red", "green"), lty = 1, lwd = 3)

c_tab_lda = table(iris$Species, pred)
misclass_lda = 1 - sum(diag(c_tab_lda)) / sum(c_tab_lda)

################################################################################

LDA = lda(Species ~ Sepal.Length + Sepal.Width, data = iris)
print(LDA)
# ------------------------------------------------------------------------------

## Task 4
# Use Models reported in 2c to generate new data of this kind with the same total
# number of cases as in the original data (hint: use sample() and rmvnorm() from
# package mvtnorm). Make a scatterplot of the same kind as in step 1 but for the
# new data and compare it with the plots for the original and the predicted data.
# Conclusions?

################################################################################
# saknar sample()

new_setosa = rmvnorm(n = 50, mean = mean_setosa, sigma = cov_setosa)
new_versicolor = rmvnorm(n = 50, mean = mean_versicolor, sigma = cov_versicolor)
new_virginica = rmvnorm(n = 50, mean = mean_virginica, sigma = cov_virginica)

plot(new_setosa[, 1], new_setosa[, 2], col = "blue", pch = 21,
     xlim = c(4.4, 7.9), ylim = c(1.9, 4.7), bg = "blue",
     main = "Sepal length vs Sepal width", xlab = "Length", ylab = "Width")
points(new_versicolor[, 1], new_versicolor[, 2],
       col = "red", pch = 21, bg = "red")
points(new_virginica[, 1], new_virginica[, 2],
       col = "green", pch = 21, bg = "green")
legend("topleft", title = "Generated",
       legend = c("Setosa", "Versicolor", "Virginica"),
       col = c("blue", "red", "green"), lty = 1, lwd = 3)

################################################################################

# ------------------------------------------------------------------------------

## Task 5
# Make a similar kind of classification by logistic regression (use function
# multinom() from nnet package), plot the classified data and compute the
# misclassification error. Compare these results with the LDA results.

log_reg = multinom(Species ~ Sepal.Length + Sepal.Width, data = iris)
iris$log_reg <- predict(log_reg, newdata = iris, "class")
c_tab_log_reg = table(iris$Species, iris$log_reg)
misclass_log_reg = 1 - sum(diag(c_tab_log_reg))/sum(c_tab_log_reg)

plot(iris$Sepal.Length, iris$Sepal.Width, pch = 21,
     bg = c("blue", "red", "green")[unclass(iris$log_reg)],
     xlab = "Length", ylab = "Width", main = "Sepal length vs Sepal width")
legend("topleft", title = "Logistic regression",
       legend = c("Setosa", "Versicolor", "Virginica"),
       col = c("blue", "red", "green"), lty = 1, lwd = 3)
