## Clear env ##
rm(list=ls())
dev.off(dev.list()["RStudioGD"])

## Import data ##
library(tree)
setwd("~/Dropbox/Skola/LiU/TDDE01/tdde01")
data <- read.csv2("./upg2_data/bank-full.csv", stringsAsFactors=TRUE)

# ------------------------------------------------------------------------------

# TASK 1 - Divide data into training, validation and test data (40/30/30) #
set.seed(12345)
src_duration <- data[,12]
data <- data[,-12]
n <- dim(data)[1]
id <- sample(1:n, floor(n*0.4))
train <- data[id,]

id1 <- setdiff(1:n, id)
set.seed(12345)
id2 <- sample(id1, floor(n*0.3))
valid <- data[id2,]

id3 <- setdiff(id1, id2)
test <- data[id3,]

# ------------------------------------------------------------------------------

# TASK 2 - Fit decision trees to data
# a) Default settings
nobs = length(train$y)
fit_default<-tree(train$y~., data=train)
#plot(fit_default)
#text(fit_default, pretty=1)
Yfit_default_train=predict(fit_default, newdata=train, type="class")
Yfit_default_valid=predict(fit_default, newdata=valid, type="class")
#table(train$y, Yfit_default_train)
#table(valid$y, Yfit_default_valid)

# b) Smallest allowed node size = 7000
control_minsize <- tree.control(nobs=nobs, minsize=7000)
fit_minsize <- tree(train$y~., data=train, control=control_minsize)
#plot(fit_minsize)
#text(fit_minsize, pretty=1)
Yfit_minsize_train=predict(fit_minsize, newdata=train, type="class")
Yfit_minsize_valid=predict(fit_minsize, newdata=valid, type="class")
#table(train$y, Yfit_minsize_train)
#table(valid$y, Yfit_minsize_valid)

# c) Smallest deviation lowered to 0.0005
control_mindev <- tree.control(nobs=nobs, mindev=0.0005)
fit_mindev <- tree(train$y~., data=train, control=control_mindev)
#plot(fit_mindev)
#text(fit_mindev, pretty=0)
Yfit_mindev_train=predict(fit_mindev, newdata=train, type="class")
Yfit_mindev_valid=predict(fit_mindev, newdata=valid, type="class")
#table(train$y, Yfit_mindev_train)
#table(valid$y, Yfit_mindev_valid)

# ------------------------------------------------------------------------------

# TASK 3 - Find optimal tree depth

trainScore=rep(0,50)
testScore=rep(0,50)
for(i in 2:50) {
  prunedTree=prune.tree(fit_mindev,best=i)
  #pred_train=predict(prunedTree, newdata=train, type="tree")
  #pred_mindev_valid=predict(prunedTree, newdata=valid, type="tree")
  pred=predict(prunedTree, newdata=valid, type="class")
  trainScore[i]=deviance(prunedTree)
}
plot(1:50, trainScore[1:50], type="b", col="red", ylim=c(10000,max(trainScore)*1.1))
#points(1:50, testScore[1:50], type="b", col="blue")

# The optimal amount of leaves is XX, and thus, these leaves are most important
# to the model. 

# Confusion matrix for test data, containing misclassification rate
# table(test$y, prune.tree(fit_mindev,best=XX))
