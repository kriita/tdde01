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

plot(iris$Sepal.Width, iris$Sepal.Length, pch = 21,
     bg=c("blue","red","green")[unclass(iris$Species)],
     xlab = "Sepal width", ylab = "Sepal length")
legend("topleft", legend = c("Setosa", "Versicolor", "Virginica"),
       col = c("blue", "red", "green"), lty = 1, lwd = 3)

# Blue is easy to classify but red and green are very mixed, so it is not easy

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
sepal = iris[,-c(3, 4, 5)]
setosa = sepal[1:50,]
versicolor = sepal[51:100,]
virginica = sepal[101:150,]

mean_width_setosa = mean(setosa$Sepal.Width)
mean_length_setosa = mean(setosa$Sepal.Length)
mean_width_versicolor = mean(versicolor$Sepal.Width)
mean_length_versicolor = mean(versicolor$Sepal.Length)
mean_width_virginica = mean(virginica$Sepal.Width)
mean_length_virginica = mean(virginica$Sepal.Length)

priorProb = 50/150 # 50 of each species, three different species and 150 samples

cov_tot = cov(sepal)
cov_setosa = cov(setosa)
cov_versicolor = cov(versicolor)
cov_virginica = cov(virginica)

# b)
pooledCov = priorProb * (cov_setosa + cov_versicolor + cov_virginica)

# c)
# p(Y = C_i | X) = exp(w_i^T * x) / sum_j=1^K exp(w_j^T * x)
# w_0i = -1/2 * mu_i^T * sigma^-1 * mu_i + log(pi_i)
# w_i = sigma^-1 * mu_i

# d)
# delta_k = x^T * w_k + w_0k
mean_setosa = as.matrix(c(mean_length_setosa, mean_width_setosa))
mean_versicolor = as.matrix(c(mean_length_versicolor, mean_width_versicolor))
mean_virginica = as.matrix(c(mean_length_virginica, mean_width_virginica))

w_setosa = inv(cov_setosa) %*% mean_setosa
w_versicolor = inv(cov_versicolor) %*% mean_versicolor
w_virginica = inv(cov_virginica) %*% mean_virginica

w0_setosa = -1/2 * t(mean_setosa) %*% w_setosa + log(priorProb)
w0_versicolor = -1/2 * t(mean_versicolor) %*% w_versicolor + log(priorProb)
w0_virginica = -1/2 * t(mean_virginica) %*% w_virginica + log(priorProb)

discr_setosa = as.matrix(sepal) %*% w_setosa + c(w0_setosa)
discr_versicolor = as.matrix(sepal) %*% w_versicolor + c(w0_versicolor)
discr_virginica = as.matrix(sepal) %*% w_virginica + c(w0_virginica)

pred_discr_setosa = max(discr_setosa)
pred_discr_versicolor = max(discr_versicolor)
pred_discr_virginica = max(discr_virginica)

## mean_tot = c(mean(iris$Sepal.Length), mean(iris$Sepal.Width))
## w_tot = inv(cov(sepal)) %*% mean_tot
## w0_tot = 1/2 * t(mean_tot) %*% w_tot + log(1)
## discr_tot = as.matrix(sepal) %*% w_tot - c(w0_tot)

# e)
# w_1 * x + w_01 = w_2 * x + w_02
# w_1 * x + w_01 = w_3 * x + w_03
# w_2 * x + w_02 = w_3 * x + w_03
# x = (length, width)

# Assumption: same covariance for each class(?)
# sort of

# ------------------------------------------------------------------------------

## Task 3
# Use discriminant functions from step 2 to predict the species from the original
# data and make a scatterplot of Sepal Length versus Sepal Width in which color
# shows the predicted Species. Estimate the misclassification rate of the prediction.
# Comment on the quality of classification. Afterwards, perform the LDA analysis
# with lda() function and investigate whether you obtain the same test error by
# using this package. Should it be same?

pred_species = 

plot(pred_species$Sepal.Length, pred_species$Sepal.Width, pch = 21,
     bg=c("blue","red","green")[unclass(pred_species$Species)],
     xlab = "Sepal width", ylab = "Sepal length")
legend("topleft", title = "Predicted species",
       legend = c("Setosa", "Versicolor", "Virginica"),
       col = c("blue", "red", "green"), lty = 1, lwd = 3)

tab = table(pred_species, iris$Species)
missclass_species = 1 - sum(diag(tab)) / length(pred_species)

LDA = lda(Species~Sepal.Length+Sepal.Width, data = iris)
print(LDA)

# It should be the same, which it also is(?)

# ------------------------------------------------------------------------------

## Task 4
# Use Models reported in 2c to generate new data of this kind with the same total
# number of cases as in the original data (hint: use sample() and rmvnorm() from
# package mvtnorm). Make a scatterplot of the same kind as in step 1 but for the
# new data and compare it with the plots for the original and the predicted data.
# Conclusions?

species = sample(iris$Species, size = 150)
gen_setosa = data.matrix(rmvnorm(n = 50,
                                 mean = c(mean_length_setosa, mean_width_setosa),
                                 sigma = cov_setosa))
gen_versicolor = data.matrix(rmvnorm(n = 50,
                                     mean = c(mean_length_versicolor, mean_width_versicolor),
                                     sigma = cov_versicolor))
gen_virginica = data.matrix(rmvnorm(n = 50,
                                    mean = c(mean_length_virginica, mean_width_virginica),
                                    sigma = cov_virginica))
generated = rbind(gen_setosa, gen_versicolor, gen_virginica)

plot(generated[, 1], generated[, 2], pch = 21,
     bg=c("blue","red","green")[unclass(species)],
     xlab = "Sepal width", ylab = "Sepal length")
legend("topleft", legend = c("Setosa", "Versicolor", "Virginica"),
       col = c("blue", "red", "green"), lty = 1, lwd = 3)

# ------------------------------------------------------------------------------

## Task 5
# Make a similar kind of classification by logistic regression (use function
# multinom() from nnet package), plot the classified data and compute the
# misclassification error. Compare these results with the LDA results.

log_reg = multinom(Species ~ Sepal.Length + Sepal.Width, newdata = iris)
plot(log_reg)
