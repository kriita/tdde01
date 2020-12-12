## Setup env ##
rm(list=ls())
dev.off(dev.list()["RStudioGD"])
library(neuralnet)

set.seed(1234567890)
Var <- runif(500, 0, 10)
mydata <- data.frame(Var, Sin=sin(Var))
tr <- mydata[1:25,] # Training
te <- mydata[26:500,] # Test
# Random initialization of the weights in the interval [-1, 1]
winit <- runif(1000,-1,1)
nn <- neuralnet(Sin ~ Var, tr, hidden=c(8,5), startweights = winit, threshold = 0.0002)
# Plot of the training data (black), test data (blue), and predictions (red)
plot(tr, main="Predicting sin(x), [0,10]", xlab="x", ylab='sin(x)', cex=2)
points(te, col = "blue", cex=1)
points(te[,1],predict(nn,te), col="red", cex=1)

# New test with high values
set.seed(1234567890)
Var <- runif(500, 0, 20)
te2 <- data.frame(Var, Sin=sin(Var))

# plot new data
plot(te2[,1],predict(nn,te2), main="Predicting sin(x), [0,20]", xlab="x", ylab='sin(x)', col="red")
points(te2[,1], te2[,2], col = "blue", cex=1)

# Predict sin
set.seed(1234567890)
X <- runif(500, 0, 10)
tr2 <- data.frame(Sin=sin(X), X)

# plot new data
nn2 <- neuralnet(X ~ Sin, tr2, hidden=c(4,3,4), startweights = winit, threshold = 2)

plot(tr2[,1], tr2[,2], main="Predicting x, [0,10]", xlab="sin(x)", ylab='x', col="blue", ylim=c(0,10))
points(tr2[,1], predict(nn2,tr2), col="red", cex=1)

