library(kknn)

# Import data from excel file. (1c:21)
data = read.csv("upg1_data/optdigits.csv", header = FALSE)

# 1. Divide data into training (50%), validation (25%) and test (25%) sets. (1e:20)
n = nrow(data)[1]
set.seed(12345)
id = sample(1:n, floor(n*0.5))
train = data[id,]

id1 = setdiff(1:n, id)
set.seed(12345)
id2 = sample(id1, floor(n*0.25))
valid = data[id2,]

id3 = setdiff(id1, id2)
test = data[id3,]

# --------------------------------------------------------------------------------------------------

# 2. Fit 30-nearest neighbour classifier. (1a:24) Documentation https://cran.r-project.org/web/packages/kknn/kknn.pdf
# V65~. means estimate column 65 from the rest of the row. 
# V65 is converted to factor because categorical variable.
model_train = kknn(as.factor(V65)~., train, train, k=30, kernel="rectangular")        # Training data
model_train_test = kknn(as.factor(V65)~., train, test, k=30, kernel="rectangular")    # Training and test data

# Confusion matrix (1c:15+16)
# Create confucion matrix of predicted values and true values (column 65). 
table(Prediction = fitted.values(model_train),TrueLables = train[,65])         # Training data
table(Prediction = fitted.values(model_train_test),TrueLables = test[,65])     # Training and test data

# Missclassification error. (1c:31)
# Sums the diagonal and divide on total number of values. 1 - this value. 
missclass = function(X, X1) {
  n = length(X)
  return(1-sum(diag(table(X, X1)))/n)
}

missclass(fitted.values(model_train), train[,65])         # Training data
missclass(fitted.values(model_train_test), test[,65])     # Training and test data

# Training data:
# Missclass: 0.04500262 (overall quality)
# The quality of preduction for different digits vary. It is not so good at 4 (0.091) or 9 (0.085).
# It is better at 0 (0.000), 2 (0.010) and 6 (0.010).

# Training and test data:
# Missclass: 0.05329154 (overall quality)
# The quality of preduction for different digits vary. It is not so good at 4 (0.138) or 8 (0.103).
# It is better at 0 (0.013), 6 (0.000) and 7 (0.009).


# --------------------------------------------------------------------------------------------------
# 3. Find "easy" and "hard" eights. 
all_eights = which(fitted.values(model_train) == 8)
prob = model_train$prob
# Column 9 because 1 indexing but digits 0-9 (--> digit 8 = index 9)
eights_prob = prob[all_eights, 9]
easy_eights = all_eights[tail(order(eights_prob), n = 2)]
hard_eights = all_eights[head(order(eights_prob), n = 3)]


# Visualize digits with heatmap.
image_easy1 <- matrix(as.matrix(train[easy_eights[1], 1:64]), nrow = 8, ncol = 8, byrow = TRUE)
heatmap(image_easy1, Rowv = NA, Colv = NA)

image_easy2 <- matrix(as.matrix(train[easy_eights[2], 1:64]), nrow = 8, ncol = 8, byrow = TRUE)
heatmap(image_easy2, Rowv = NA, Colv = NA)

image_hard1 <- matrix(as.matrix(train[hard_eights[1], 1:64]), nrow = 8, ncol = 8, byrow = TRUE)
heatmap(image_hard1, Rowv = NA, Colv = NA)

image_hard2 <- matrix(as.matrix(train[hard_eights[2], 1:64]), nrow = 8, ncol = 8, byrow = TRUE)
heatmap(image_hard2, Rowv = NA, Colv = NA)

image_hard3 <- matrix(as.matrix(train[hard_eights[3], 1:64]), nrow = 8, ncol = 8, byrow = TRUE)
heatmap(image_hard3, Rowv = NA, Colv = NA)

# The two "easy" eights were not so easy to classify according to me, but you can see that it is an eight. 
# The three hard cases are all missclassified, they are one 7 and two 9. 
# So it makes sense that they had a low probability of being an eight. 

# Note: Lab assistant was okey with pictures being upside down. 

# --------------------------------------------------------------------------------------------------
# 4. Fit 1:30-nearest neigbour classifier and plot the dependence of the 
# training and validation misclassification errors on the value of K.

# Missclass for training data for k = 1:30
missclass_train = rep(x = 0, times = 30)
for (i in seq(1:30)) {
  pred = kknn(as.factor(V65)~., train, train, k=i, kernel="rectangular")
  missclass_train[i] = missclass(fitted.values(pred), train[,65])
}

# Missclass for validation data for k = 1:30
missclass_valid = rep(x = 0, times = 30)
for (i in seq(1:30)) {
  pred = kknn(as.factor(V65)~., train, valid, k=i, kernel="rectangular")
  missclass_valid[i] = missclass(fitted.values(pred), valid[,65])
}


# From https://daviddalpiaz.github.io/r4sl/knn-class.html#categorical-data
# Plot missclass rate for training and validation sets. 
plot(missclass_train, type = "b", col = "red", cex = 1, pch = 20, 
     xlab = "k, number of neighbors", ylab = "missclass error",
     main = "Missclass Rate vs Neighbors", ylim = c(0, 0.06))
lines(missclass_valid, type = "b", col = "blue")
legend("topleft", legend=c("Training", "Validation"), col=c("red", "blue"), lty=1, cex=1)


# Larger K means less complex bounderies (less sharp edges). See https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fi1.wp.com%2Fscikit-learn.org%2Fstable%2F_images%2Fsphx_glr_plot_classification_0011.png%3Fresize%3D690%252C518&f=1&nofb=1 
# Smaller K means high variance and low bias = overfitting and high complexity. (1e:22) and https://www.geeksforgeeks.org/knn-model-complexity/ 
# Optimal K = 7 becasue after that point the missclass error starts rising. 

# # Missclass for test data with optimal k = 7
pred = kknn(as.factor(V65)~., train, test, k=7, kernel="rectangular")
missclass_test = missclass(fitted.values(pred), test[,65])

# Add test missclass to plot
points(7, missclass_test,  type = "b", col = "green")
legend("topleft", legend=c("Training", "Validation", "Test"), col=c("red", "blue", "green"), lty=1, cex=1)

# --------------------------------------------------------------------------------------------------
# 5. Emperical risk and cross-entropy for k = 1:30
# Cross-entropy: https://rpubs.com/juanhklopper/cross_entropy
# Cross-entropy loss: https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html

cross_entropy_loss = function(true_labels, pred){
  x = 0
  for (i in 1:length(true_labels)) { 
    for (n in 1:10) {
      if (true_labels[i] == (n - 1)) {
        x = x + log(pred$prob[i, n]+ exp(-15)) # Add exp(-15) for numerical reason. 
      }
    }
  }
  return(-x)
}
# Calculate cross-entropy loss for training data.
cross_entr_train = rep(x = 0, times = 30)
for (i in seq(1:30)) {
  pred = kknn(as.factor(V65)~., train, train, k=i, kernel="rectangular")
  cross_entr_train[i] = cross_entropy_loss(train[,65], pred)
}

# Calculate cross-entropy loss for validation data.
cross_entr_valid = rep(x = 0, times = 30)
for (i in seq(1:30)) {
  pred = kknn(as.factor(V65)~., train, valid, k=i, kernel="rectangular")
  cross_entr_valid[i] = cross_entropy_loss(valid[,65], pred)
}

# Plot cross-entropy loss for the different k:s. 
plot(cross_entr_train, type = "b", col = "red", cex = 1, pch = 20, 
     xlab = "k, number of neighbors", ylab = "cross-entropy loss",
     main = "Cross-entropy Loss vs Neighbors", ylim = c(0, 400))
lines(cross_entr_valid, type = "b", col = "blue")
legend("topleft", legend=c("Training", "Validation"), col=c("red", "blue"), lty=1, cex=1)


# Optimal k = 6
# Cross-entropy is more suitable than missclassification because it contains more information 
# about the predictions. If a prediction has a high probability but is missclassified, it will 
# be punished. And if a prediction has a low probability and is classified correctly, it will 
# not gain so much. In this way we can more correctly compare different classifiers on how 
# certain they were when classifieng an observation, which is not possible when using missclassification
# rate. 


