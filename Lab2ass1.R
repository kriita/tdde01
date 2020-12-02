require(ggplot2)
require(MASS)
require(matlib)
require(glmnet)
require(mvtnorm)
require(nnet)

attach(iris) # put the iris data in your current work directory

## Task 1
# Make a scatterplot of Sepal Width versus Sepal Length where observations are
# colored by Species.

# take out only interesting data
# x = (length width)
sepal = iris[,-c(3, 4, 5)]
setosa = sepal[1:50,]
versicolor = sepal[51:100,]
virginica = sepal[101:150,]

plot(iris$Sepal.Length, iris$Sepal.Width, pch = 21,
     col = c("blue", "red", "green")[unclass(iris$Species)],
     bg = c("blue", "red", "green")[unclass(iris$Species)],
     main = "Sepal length vs Sepal width", xlab = "Length", ylab = "Width")
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
mean_setosa = as.matrix(c(mean(setosa[, 1]), mean(setosa[, 2])))
mean_versicolor = as.matrix(c(mean(versicolor[, 1]), mean(versicolor[, 2])))
mean_virginica = as.matrix(c(mean(virginica[, 1]), mean(virginica[, 2])))

priorProb = 50/150 # 50 of each species, three different species and 150 samples

cov_setosa = cov(setosa)
cov_versicolor = cov(versicolor)
cov_virginica = cov(virginica)

# b)
pooledCov = priorProb * (cov_setosa + cov_versicolor + cov_virginica)

# c)
# x|y = C_i, mu_i, Sigma ~ N(mu_i, Sigma)
# y|pi ~ Multinomial(pi_1, ..., pi_k)

# d)
# w_0i = -1/2 * mu_i^T * Sigma^-1 * mu_i + log(pi_i)
# w_i = Sigma^-1 * mu_i
# delta_k = x^T * w_k + w_0k
w_setosa = as.matrix(inv(pooledCov) %*% mean_setosa)
w_versicolor = as.matrix(inv(pooledCov) %*% mean_versicolor)
w_virginica = as.matrix(inv(pooledCov) %*% mean_virginica)

w0_setosa = -1/2 * t(mean_setosa) %*% w_setosa + log(priorProb)
w0_versicolor = -1/2 * t(mean_versicolor) %*% w_versicolor + log(priorProb)
w0_virginica = -1/2 * t(mean_virginica) %*% w_virginica + log(priorProb)

discr_setosa = as.matrix(sepal) %*% w_setosa + c(w0_setosa)
discr_versicolor = as.matrix(sepal) %*% w_versicolor + c(w0_versicolor)
discr_virginica = as.matrix(sepal) %*% w_virginica + c(w0_virginica)

# e)
# (w_1 - w_2)_l * x + (w_1 - w_2)_w * y + (w0_1 - w0_2) = 0
# (w_1 - w_3)_l * x + (w_1 - w_3)_w * y + (w0_1 - w0_3) = 0
# (w_2 - w_3)_l * x + (w_2 - w_3)_w * y + (w0_2 - w0_3) = 0

# ------------------------------------------------------------------------------

## Task 3
# Use discriminant functions from step 2 to predict the species from the original
# data and make a scatterplot of Sepal Length versus Sepal Width in which color
# shows the predicted Species. Estimate the misclassification rate of the
# prediction. Afterwards, perform the LDA analysis with lda() function and
# investigate whether you obtain the same test error by using this package.

# initialize matrices
pred_setosa = cbind(c(), c())
pred_versicolor = cbind(c(), c())
pred_virginica = cbind(c(), c())
pred = c()

# compare each discriminant value and add corresponding sepal length/width to
# the class with greater value
for (i in c(1:150)) {
        if (discr_setosa[i] > discr_versicolor[i]) {
                if (discr_setosa[i] > discr_virginica[i]) {
                        pred_setosa <- cbind(c(pred_setosa[, 1], sepal[i, 1]),
                                             c(pred_setosa[, 2], sepal[i, 2]))
                        pred <- c(pred, "setosa")
                } else {
                        pred_virginica <- cbind(c(pred_virginica[, 1], sepal[i, 1]),
                                                c(pred_virginicar[, 2], sepal[i, 2]))
                        pred <- c(pred, "virginica")
                }
        } else {
                if (discr_versicolor[i] > discr_virginica[i]) {
                        pred_versicolor <- cbind(c(pred_versicolor[, 1], sepal[i, 1]),
                                                 c(pred_versicolor[, 2], sepal[i, 2]))
                        pred <- c(pred, "versicolor")
                } else {
                        pred_virginica <- cbind(c(pred_virginica[, 1], sepal[i, 1]),
                                                 c(pred_virginica[, 2], sepal[i, 2]))
                        pred <- c(pred, "virginica")
                }
        }
}

plot(pred_setosa[, 1], pred_setosa[, 2],
     pch = 21, col = "blue", bg = "blue",
     xlab = "Length", ylab = "Width", main = "Sepal length vs Sepal width",
     xlim = c(4.3, 7.9), ylim = c(2, 4.4))
points(pred_versicolor, pch = 21, col = "red", bg = "red")
points(pred_virginica, pch = 21, col = "green", bg = "green")
legend("topleft", title = "Predicted",
       legend = c("Setosa", "Versicolor", "Virginica"),
       col = c("blue", "red", "green"), lty = 1, lwd = 3)

iris$pred <- pred
c_tab_discr = table(iris$Species, iris$pred)
misclass_discr = 1 - sum(diag(c_tab_discr))/sum(c_tab_discr)

LDA = lda(Species ~ Sepal.Length + Sepal.Width, data = iris)
print(LDA)
pred_lda = predict(LDA, iris)
c_tab_lda = table(iris$Species, pred_lda$class)
misclass_lda = 1 - sum(diag(c_tab_lda))/sum(c_tab_lda)

# ------------------------------------------------------------------------------

# HAR EJ ANVÄNT SAMPLE() I TASK 4, VILKET VERKAR VARA ETT KRAV

## Task 4
# Use models reported in 2c to generate new data of this kind with the same total
# number of cases as in the original data (hint: use sample() and rmvnorm() from
# package mvtnorm). Make a scatterplot of the same kind as in step 1 but for the
# new data

new_setosa = rmvnorm(n = 50, mean = mean_setosa, sigma = cov_setosa)
new_versicolor = rmvnorm(n = 50, mean = mean_versicolor, sigma = cov_versicolor)
new_virginica = rmvnorm(n = 50, mean = mean_virginica, sigma = cov_virginica)

iris$gen_length <- c(new_setosa[, 1], new_versicolor[, 1], new_virginica[, 1])
iris$gen_width <- c(new_setosa[, 2], new_versicolor[, 2], new_virginica[, 2])

plot(iris$gen_length, iris$gen_width, pch = 21,
     col = c("blue", "red", "green")[unclass(iris$Species)],
     bg = c("blue", "red", "green")[unclass(iris$Species)],
     main = "Sepal length vs Sepal width", xlab = "Length", ylab = "Width")
legend("topleft", title = "Generated",
       legend = c("Setosa", "Versicolor", "Virginica"),
       col = c("blue", "red", "green"), lty = 1, lwd = 3)

# ------------------------------------------------------------------------------

## Task 5
# Make a similar kind of classification by logistic regression (use function
# multinom() from nnet package), plot the classified data and compute the
# misclassification error.

log_reg = multinom(Species ~ Sepal.Length + Sepal.Width, data = iris)
iris$log_reg <- predict(log_reg, newdata = iris, "class")
c_tab_log_reg = table(iris$Species, iris$log_reg)
misclass_log_reg = 1 - sum(diag(c_tab_log_reg))/sum(c_tab_log_reg)

plot(iris$Sepal.Length, iris$Sepal.Width, pch = 21,
     col = c("blue", "red", "green")[unclass(iris$log_reg)],
     bg = c("blue", "red", "green")[unclass(iris$log_reg)],
     xlab = "Length", ylab = "Width", main = "Sepal length vs Sepal width")
legend("topleft", title = "Logistic regression",
       legend = c("Setosa", "Versicolor", "Virginica"),
       col = c("blue", "red", "green"), lty = 1, lwd = 3)