## Binary classification using LOGISTIC REGRESSION
## Using Gradient Descent (GD) and Stochastic Gradient Descent (SGD)

rm(list = ls())

## Import dataset and training parameters
train <- read.table("hw3_train.dat")
test <- read.table("hw3_test.dat")

Xtrain <- train[, 1:ncol(train)-1]
Xtrain <- as.matrix(Xtrain)
Xtrain <- cbind(rep(1, nrow(Xtrain)), Xtrain)     # (Nxd) matrix
ytrain <- train[, ncol(train)]
ytrain <- as.numeric(ytrain)                      # (Nx1) vector
w <- rep(0, ncol(Xtrain))          # initial weight (dx1) vector
N <- nrow(Xtrain)                  # number of examples
eta <- .001                        # learning rate



## Part 1: Gradient Descent for Logistic Regression
library(pracma)  # for sigmoid fcn
Earr <- c()
for (i in 1:2000) {
    grad <- (t(- ytrain * Xtrain) %*% sigmoid(- ytrain * (Xtrain %*% w)))/ N  # (dxN) * (Nx1) = (dx1) gradient
    w <- w - eta * grad
    Ece <- mean(-log(sigmoid(ytrain * (Xtrain %*% w))))  # cross-entropy error (Cost fcn)
    Earr <- c(Earr, Ece)
}
y_pred <- sigmoid(Xtrain %*% w)
y_pred[y_pred >= 0.5] <- 1
y_pred[y_pred < 0.5] <- -1
Ein <- mean(as.numeric(y_pred != ytrain))



## Part 2: Stochastic Gradient Descent for Logistic Regression
library(pracma)  # for sigmoid fcn
ws <- rep(0, ncol(Xtrain))
Esarr <- c()
for (i in 1:2000) {
    order <- i %% N
    if (order == 0) {
        order <- N
    }
    grad <- as.numeric((- ytrain * Xtrain)[order, ]) * sigmoid(- ytrain * (Xtrain %*% ws))[order]  # (dx1) * (1x1) = (dx1) gradient
    ws <- ws - eta * grad
    Ece <- mean(-log(sigmoid(ytrain * (Xtrain %*% ws))))  # cross-entropy error (Cost fcn)
    Esarr <- c(Esarr, Ece)
}
y_pred <- sigmoid(Xtrain %*% w)
y_pred[y_pred >= 0.5] <- 1
y_pred[y_pred < 0.5] <- -1
Esin <- mean(as.numeric(y_pred != ytrain))



## Testing
Xtest <- test[, 1:ncol(test)-1]
Xtest <- as.matrix(Xtest)
Xtest <- cbind(rep(1, nrow(Xtest)), Xtest)          # (Nxd) matrix
ytest <- test[, ncol(test)]
ytest <- as.numeric(ytest)                          # (Nx1) vector

  # Gradient Descent testing
y_pred <- sigmoid(Xtest %*% w)
y_pred[y_pred >= 0.5] <- 1
y_pred[y_pred < 0.5] <- -1
Eout <- mean(as.numeric(y_pred != ytest))

  # Stochastic Gradient Descent testing
y_pred <- sigmoid(Xtest %*% ws)
y_pred[y_pred >= 0.5] <- 1
y_pred[y_pred < 0.5] <- -1
Esout <- mean(as.numeric(y_pred != ytest))

