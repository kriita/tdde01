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
plot(tr, cex=2)
points(te, col = "blue", cex=1)
points(te[,1],predict(nn,te), col="red", cex=1)

# New test with high values
set.seed(1234567890)
Var <- runif(500, 0, 20)
te2 <- data.frame(Var, Sin=sin(Var))

# plot new data
plot(te2[,1],predict(nn,te2), main="boop", col="red")
points(te2[,1], te2[,2], col = "blue", cex=1)

# Predict sin
set.seed(1234567890)
X <- runif(500, 0, 20)
tr2 <- data.frame(Sin=sin(X), X)

# plot new data
nn2 <- neuralnet(X ~ Sin, tr2, hidden=c(8,5), startweights = winit, threshold = 0.02)

plot(te2[,1], predict(nn2,te2), main="boop2", col="red", ylim=c(-1.2,12))
points(te2[,1], te2[,2], col="blue", cex=1)
