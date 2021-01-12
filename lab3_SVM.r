# Lab 3 block 1 of 732A99/TDDE01 Machine Learning
# Author: jose.m.pena@liu.se
# Made for teaching purposes

library(kernlab)
set.seed(1234567890)

data(spam)

index <- sample(1:4601)
tr <- spam[index[1:3000], ]
va <- spam[index[3001:3800], ]
trva <- spam[index[1:3800], ]
te <- spam[index[3801:4601], ]
by <- 0.3
err_va <- NULL

for(i in seq(by,5,by)){
  filter <- ksvm(type~.,data=tr,kernel="rbfdot",kpar=list(sigma=0.05),C=i)
  mailtype <- predict(filter,va[,-58])
  t <- table(mailtype,va[,58])
  err_va <- c(err_va,(t[1,2]+t[2,1])/sum(t))
}

filter0 <- ksvm(type~.,data=tr,kernel="rbfdot",kpar=list(sigma=0.05),
                C=which.min(err_va)*by)
mailtype0 <- predict(filter0,va[,-58])
t0 <- table(mailtype0,va[,58])
err0 <- (t0[1,2]+t0[2,1])/sum(t0)
err0

filter1 <- ksvm(type~.,data=tr,kernel="rbfdot",kpar=list(sigma=0.05),
                C=which.min(err_va)*by)
mailtype1 <- predict(filter1,te[,-58])
t1 <- table(mailtype1,te[,58])
err1 <- (t1[1,2]+t1[2,1])/sum(t1)
err1

filter2 <- ksvm(type~.,data=trva,kernel="rbfdot",kpar=list(sigma=0.05),
                C=which.min(err_va)*by)
mailtype2 <- predict(filter2,te[,-58])
t2 <- table(mailtype2,te[,58])
err2 <- (t2[1,2]+t2[2,1])/sum(t2)
err2

filter3 <- ksvm(type~.,data=spam,kernel="rbfdot",
                kpar=list(sigma=0.05),C=which.min(err_va)*by)
mailtype3 <- predict(filter3,te[,-58])
t3 <- table(mailtype3,te[,58])
err3 <- (t3[1,2]+t3[2,1])/sum(t3)
err3

# Questions

# 1. Which model do we return to the user: filter0, filter1, filter2 or filter3?
# Why?

# 2. What is the estimate of the generalization error of the model selected:
# err0, err1, err2 or err3? Why?
# unseen data -> not err3
