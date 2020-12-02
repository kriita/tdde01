library(pls)
library(ggplot2)
library(boot)

# Set seed :)
set.seed(12345)

# Import data from excel file. (1c:21)
data = read.csv("upg2_data/communities.csv")

# 1. Scale variables except ViolentCrimesPerPop, and implement PCA.
scaled_data = scale(data[,-c(101)])

# Calculate covariance matrix. Tells us how each variable correlates to each other. 
covariance = cov(scaled_data)

# Calculate eigen values and eigen vectors.
eigen = eigen(covariance)

# proportion of variation
sprintf("%2.3f", eigen$values/sum(eigen$values)*100)

# Empirical Cumulative Distribution Function
#variance_explained = ecdf(eigen$values/sum(eigen$values))

cs = cumsum(eigen$values/sum(eigen$values))

# x-axis is #PC variables
plot(cs, type = "l")
abline(h = 0.95, col = "red")

num_vars_needed = head(which(cs >= 0.95), n = 1)
num_vars_needed
# What is the proportion of variation
# explained by each of the first two principal components?
head(eigen$values/sum(eigen$values)*100, n = 2)

# 2. princomp()

pca = princomp(scaled_data)

plot(pca$loadings[,1], type = "l", xlab = "features", ylab = "effect on PC1", main = "First Principal Component")


# Do many features have a notable contribution to this component?

# Many features affects the component, but they are not so big, very close to 0.
# No feature have a major impact by itself. 

top_five_features = pca$loadings[, 1][tail(order(abs(pca$loadings[, 1])), n = 5)]
# Comment whether these features have anything in common and whether they may have a logical relationship to the crime level
# Logical. These are incorporated in PC1, which is logical. 


# Plot of PC scores, PC1 and PC2

pc_scores = as.data.frame(pca$scores[, 1:2])

ViolentCrimesPerPop = data$ViolentCrimesPerPop
ggplot(pc_scores, aes(x=Comp.1, y=Comp.2, color = ViolentCrimesPerPop)) + 
  geom_point(size = 0.5) + scale_color_gradient2(mid = "blue", high = "red")

# x-axis is the economy and child-axis. (Top five features). pos = less income, no kids, poverty. 
# y-axis is forgein born an immigration- axis. Down is more immigrated population. 
# like an triange. If high income not so much variance in immigration. And less income is more variance. 

# 3. 

#pc1 = 


lm_data = data.frame(Features = poly(pca$scores[, 1], degree = 2, raw = TRUE), Target = ViolentCrimesPerPop)

lm_fit = lm(Target~., data = lm_data)
plot(pca$scores[, 1], ViolentCrimesPerPop, cex = 0.1, xlab = "PC1 scores", main = "PC1 scores vs ViolentCrimesPerPop")
points(pca$scores[, 1], lm_fit$fitted.values, col = "red", cex = 0.1)
legend("topleft", legend=c("Predicted"), col=c("red"), lty=1, cex=0.5)

# Can the target be well explained by this feature? 

# Not so good, the variance in data is high. Hard to estimate from this. For some x it is okay, but for some very bad. 
# 5-10 (pc score) is hard to predict, very big variance in data. 
# The red line is in the middle (mean value), the best we can do. 

# Does the model seem to capture the connection between the target and the feature?

# The line (model) follows the feature quite good. 
# But it has a great variance, so it is just as good as possible. 


#4. Parametric Bootstrap
# Helper for geneterating from underlying distribution. 
rng = function(lm_data, lm_fit) {
  data1 = data.frame(Target = lm_data$Target, Features.1 = lm_data$Features.1, Features.2 = lm_data$Features.2)
  n = length(data1$Target)
  # Generate new Target
  data1$Target = rnorm(n, predict(lm_fit, newdata = data1), sd(lm_fit$residuals))
  return(data1)
}

boot_conf_band = function(data1) {
  res = lm(Target~., data = data1)
  pred = predict(res, newdata = lm_data)
  return(pred)
}

boot_pred_band = function(data1) {
  res = lm(Target~., data = data1)
  pred = predict(res, newdata = lm_data)
  return(rnorm(length(data1$Target), pred, sd(lm_fit$residuals)))
}

# Calculate confidence band
res_conf = boot(lm_data, statistic = boot_conf_band, ran.gen = rng, mle = lm_fit, R=1000, sim = "parametric")
conf_band = envelope(res_conf)

# Calculate prediction band
res_pred = boot(lm_data, statistic = boot_pred_band, ran.gen = rng, mle = lm_fit, R=1000, sim = "parametric")
pred_band = envelope(res_pred)


plot(pca$scores[, 1], lm_data$Target, col = "black", xlab = "PC1 scores", ylab = "ViolentCrimesPerPop", main = "PC1 scores vs ViolentCrimesPerPop", cex = 0.1)
points(pca$scores[, 1], predict(lm_fit, newdata = lm_data), col = "red", cex = 0.1)
points(pca$scores[, 1], conf_band$point[2,], col = "green", cex = 0.1)
points(pca$scores[, 1], conf_band$point[1,], col = "green", cex = 0.1)
points(pca$scores[, 1], pred_band$point[2,], col = "purple", cex = 0.2)
points(pca$scores[, 1], pred_band$point[1,], col = "purple", cex = 0.2)

legend("topleft", legend=c("Orignal", "Predicted", "Confidence", "Predictionband"),
       col=c("black", "red", "green", "purple"), lty=1, cex=0.5)



# What can be concluded by looking at a) confidence intervals b) prediction intervals? 

# a) Small conf interval which means that we are sure about our estimation. 
# Much data which makes us sure about this is the best line. (does not mean it will predict the correct value)

# b) Large pred interval because the data for the y-value is so spread, large variance. 
# Many possible y-values for the same x-value, which makes it hard to predict. 

# it will not predict so good. 








