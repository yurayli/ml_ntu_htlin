## N = 1000, X belongs [-1,1]x[-1,1], f = sign(x1^2 + x2^2 - 0.6)
## Add 10% noise, so y = f + noise
## Use Linear Regression to implement Classification
## Use different features and run 1000 times each tracing error

rm(list = ls())
## Part 1: Use features (1, x1, x2) only
Einarr <- c()
X <- matrix(c(runif(1000, -1, 1), runif(1000, -1, 1)), 1000, 2)
X <- cbind(rep(1, 1000), X)
f <- rep(0, 1000)
for (i in 1:1000) {
    f[i] <- sign(X[i,2]^2 + X[i,3]^2 - 0.6)
}
for (t in 1:1000) {
    ## Setup
    noise <- sample(1000, 1000*0.1)
    y <- f; y[noise] <- -y[noise]
    
    ## Linear regression
    library(pracma)
    # w <- solve(t(X) %*% X) %*% t(X) %*% y   
    w <- pinv(X) %*% y
    
    y_pred <- sign(X %*% w)
    Ein <- mean(as.numeric(y != y_pred))
    Einarr <- c(Einarr, Ein)
}
mean(Einarr)


## Part 2: Use polynomial features: (1, x1, x2, x1^2, x1x2, x2^2)
Earr <- c()
for (t in 1:1000) {
    ## Setup
    X <- matrix(c(runif(1000, -1, 1), runif(1000, -1, 1)), 1000, 2)
    X <- cbind(rep(1, 1000), X)
    f <- rep(0, 1000)
    for (i in 1:1000) {
        f[i] <- sign(X[i,2]^2 + X[i,3]^2 - 0.6)
    }
    noise <- sample(1000, 1000*0.1)
    y <- f; y[noise] <- -y[noise]
    
    ## Create polynomial features
    newX <- rep(1, 1000)
    for (i in 1:2) {
        for (j in 0:i) {
            newX <- cbind(newX, X[, 2]^(i-j) * X[, 3]^j)
        }
    }
    
    ## Linear regression
    library(pracma)
    # w <- solve(t(X) %*% X) %*% t(X) %*% y   
    w_tilde <- pinv(newX) %*% y
    
    y_pred <- sign(newX %*% w_tilde)
    E <- mean(as.numeric(y != y_pred))
    Earr <- c(Earr, E)
}
mean(Earr)

