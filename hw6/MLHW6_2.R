## Implement kernel ridge regression algorithm for classification (LSSVM)

rm(list = ls())
# Import data
dat <- read.table("hw6_lssvm_all.dat")
train <- dat[1:400,]
test <- dat[401:500,]

# Model parameters
gamm <- c(32, 2, .125) # for the Gaussian kernel
lambda <- c(.001, 1, 1000) # for L2-regularization parameter

# Training and testing
Einarr <- c()
EinMin <- 1
Eoutarr <- c()
EoutMin <- 1
for (gamm in gamm) {
    for (lambda in lambda) {
        # Training
         # Kernel matrix
        X <- as.matrix(train[,1:(ncol(train)-1)])
        N <- nrow(train)
        K <- matrix(rep(0, N), N, N)
        for (i in 1:N) {
            for (j in 1:N) {
                K[i,j] <- exp(- gamm * (X[i,] - X[j,]) %*% (X[i,] - X[j,]))
            }
        }
        beta <- solve(lambda * diag(N) + K) %*% train[,ncol(train)]
        ypred <- sign(t(beta) %*% K)
        Ein <- mean(as.numeric(ypred != train[,ncol(train)]))
        Einarr <- c(Einarr,Ein)
        if (Ein < EinMin) {
            EinMin <- Ein
            g_in <- gamm
            lam_in <- lambda
        }
        
        # Testing
        Xtest <- as.matrix(test[,1:ncol(test)-1])
        Ntest <- nrow(test)
        K <- matrix(rep(0, N), N, Ntest)
        for (i in 1:N) {
            for (j in 1:Ntest) {
                K[i,j] <- exp(- gamm * (X[i,] - Xtest[j,]) %*% (X[i,] - Xtest[j,]))
            }
        }
        ypred <- sign(t(beta) %*% K)
        Eout <- mean(as.numeric(ypred != test[,ncol(test)]))
        Eoutarr <- c(Eoutarr,Eout)
        if (Eout < EoutMin) {
            EoutMin <- Eout
            g_out <- gamm
            lam_out <- lambda
        }
    }
}

rm(i, j, Ein, Eout)