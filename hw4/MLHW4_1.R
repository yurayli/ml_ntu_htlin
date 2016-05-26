## Experiment with Regularized Linear Regression and Validation
## Learning problem for binary classification using linear regression

rm(list = ls())

## Import dataset
train <- read.table("hw4_train.dat")
test <- read.table("hw4_test.dat")

X <- train[, 1:ncol(train)-1]; X <- as.matrix(X)
X <- cbind(rep(1, nrow(X)), X)       					# (Nxd) matrix
y <- train[, ncol(train)]; y <- as.matrix(y)        	# (Nx1) vector
Xtest <- test[, 1:ncol(test)-1]; Xtest <- as.matrix(Xtest)
Xtest <- cbind(rep(1, nrow(Xtest)), Xtest)          	# (Nxd) matrix
ytest <- test[, ncol(test)]; ytest <- as.matrix(ytest)  # (Nx1) vector
#lambda <- 10                                       	# regularization factor
lambda <- c(1e2, 1e1, 1e0, 1e-1, 1e-2, 1e-3, 1e4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10)

## Regularized linear regression for classification
Einarr <- c()
Eoutarr <- c()
library(pracma)
for (k in 1:length(lambda)) {
    ## regularization factor for analytic solution
    ## w = (X_train^T * X_train + lambda * I)^-1 * (X_train^T * y)
    ##   = (X^T * X)^-1 * X^T * y
    ##   = pinv(X) * y
    ## where X = [X], y = [y]
    ##           [Xtild ]      [ytild ]
    ##       Xtild = sqrt(lambda) * I                 	 (dxd) matrix
    ##       ytild = 0                                   (dx1) vector
    Xtild <- sqrt(lambda[k]) * diag(ncol(X))    # where regularization factor
    ytild <- rep(0, ncol(X))
    X <- rbind(X, Xtild)
    y <- c(y, ytild)
    # w <- solve(t(X) %*% X) %*% t(X) %*% y   
    w <- pinv(X) %*% y
    y_pred <- sign(X %*% w)
    Ein <- mean(as.numeric(y != y_pred))
    Einarr <- c(Einarr, Ein)
    
    # Testing
    y_pred <- sign(Xtest %*% w)
    Eout <- mean(as.numeric(y_pred != ytest))
    Eoutarr <- c(Eoutarr, Eout)
}


## ----------------------------------------------------------------------------
## Validation
Xtra <- X[1:120,]
ytra <- y[1:120]
Xval <- X[121:nrow(X),]
yval <- y[121:length(y)]
Etraarr <- c()
Evalarr <- c()
Eoutarr <- c()
for (k in 1:length(lambda)) {
    Xtild <- sqrt(lambda[k]) * diag(ncol(Xtra))     # where regularization factor
    ytild <- rep(0, ncol(Xtra))
    X <- rbind(Xtra, Xtild)
    y <- c(ytra, ytild)
    w <- pinv(X) %*% y
    y_pred <- sign(Xtra %*% w)
    Etra <- mean(as.numeric(ytra != y_pred))
    Etraarr <- c(Etraarr, Etra)
    
    # Validating
    y_pred <- sign(Xval %*% w)
    Eval <- mean(as.numeric(y_pred != yval))
    Evalarr <- c(Evalarr, Eval)
    
    # Testing
    y_pred <- sign(Xtest %*% w)
    Eout <- mean(as.numeric(y_pred != ytest))
    Eoutarr <- c(Eoutarr, Eout)
}

# Training again with picked lambda
minval <- which(Evalarr == min(Evalarr))[1]
Xtild <- sqrt(lambda[minval]) * diag(ncol(X))    # where regularization factor
ytild <- rep(0, ncol(X))
X <- rbind(X, Xtild)
y <- c(y, ytild)
w <- pinv(X) %*% y
y_pred <- sign(X %*% w)
Ein <- mean(as.numeric(y != y_pred))

# Testing again
y_pred <- sign(Xtest %*% w)
Eout <- mean(as.numeric(y_pred != ytest))


## ----------------------------------------------------------------------------
## Cross validation
Ecvarr <- c()
for (k in 1:length(lambda)) {
    Evalarr <- c()
    for (v in 1:5) {
        Xval <- X[(1+40*(v-1)) : (40*v),]
        yval <- y[(1+40*(v-1)) : (40*v)]
        if (v == 1) {
            Xtra <- X[(1+40*v):200,]
            ytra <- y[(1+40*v):200]
        } else if (v == 5) {
            Xtra <- X[1:(40*(v-1)),]
            ytra <- y[1:(40*(v-1))]
        } else {
            Xtra <- rbind(X[1:(40*(v-1)),], X[(1+40*v):200,])
            ytra <- c(y[1:(40*(v-1))], y[(1+40*v):200])
        }
        
        # Regression training
        Xtild <- sqrt(lambda[k]) * diag(ncol(Xtra))    # where regularization factor
        ytild <- rep(0, ncol(Xtra))
        X <- rbind(Xtra, Xtild)
        y <- c(ytra, ytild)
        w <- pinv(X) %*% y
        
        # Validating
        y_pred <- sign(Xval %*% w)
        Eval <- mean(as.numeric(y_pred != yval))
        Evalarr <- c(Evalarr, Eval)
    }
    Ecv <- mean(Evalarr)
    Ecvarr <- c(Ecvarr, Ecv)
    
    
}

# Training again with picked lambda
minval <- which(Ecvarr == min(Ecvarr))[1]
Xtild <- sqrt(lambda[minval]) * diag(ncol(X))    # where regularization factor
ytild <- rep(0, ncol(X))
X <- rbind(X, Xtild)
y <- c(y, ytild)
w <- pinv(X) %*% y
y_pred <- sign(X %*% w)
Ein <- mean(as.numeric(y != y_pred))

# Testing again
y_pred <- sign(Xtest %*% w)
Eout <- mean(as.numeric(y_pred != ytest))
