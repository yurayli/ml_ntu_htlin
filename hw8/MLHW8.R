## Implement Artificial Neural Network
## For classification problem

rm(list = ls())

# Import dataset
train <- read.table("hw4_nnet_train.dat")
test <- read.table("hw4_nnet_test.dat")

# Plot training data
plot(train[train$V3 == 1, 1], train[train$V3 == 1, 2], type = 'p', col = 'blue',
     lwd = 2, xlab = 'x1', ylab = 'x2', xlim = c(-1,1), ylim = c(-1,1))
points(train[train$V3 == -1, 1], train[train$V3 == -1, 2], col = 'red',
     pch = 4, lwd = 2, xlab = 'x1', ylab = 'x2', xlim = c(-1,1), ylim = c(-1,1))

# Data parameters
N <- nrow(train)                                  # num of examples
d <- ncol(train)-1                                # num of features
X <- train[, 1:d]; X <- as.matrix(X)
X <- cbind(rep(1, N), X)                          # (Nx(d+1)) array
Xtest <- test[, 1:d]; Xtest <- as.matrix(Xtest)
Xtest <- cbind(rep(1, nrow(Xtest)), Xtest)        # (Nx(d+1)) array
y <- train[, d+1]; y <- as.matrix(y)              # (Nx1) array
ytest <- test[, d+1]; ytest <- as.matrix(ytest)   # (Nx1) array


part <- 3

## Part 1: test convergence of ANN
if (part == 1) {
    # Training parameters
    M <- 3                                            # d-M-1 ANN, M in c(1,6,11,16,21)
    eta <- .1                                         # d-M-1 ANN, eta in c(.001,.01,.1,1,10)
    w1 <- runif((d+1)*M, -.1, .1)                     # randomly (uniformly) initialized weight
    w2 <- runif((M+1)*1, -.1, .1)
    
    # Backprop algorithm
    Esq_arr <- c()
    ptm <- proc.time()
    for (iter in 1:5e4) {
        
        # Step 1: feedforward and calculate SQUARED ERROR measure
        w1 <- matrix(w1, (d+1), M)                        # ((d+1)xM) array
        w2 <- matrix(w2, (M+1), 1)                        # ((M+1)x1) array
        hidden <- tanh(X %*% w1)
        hidden <- cbind(rep(1, N), hidden)                # (Nx(M+1)) array
        H <- tanh(hidden %*% w2)                          # hypothesis (Nx1) array
        Esq <- mean((y - H)^2)
        Esq_arr <- c(Esq_arr, Esq)
        # Step 2: backprop and get the gradie nt of SQUARED ERROR
        delH <- -2 * (y - H) * (1 - H^2)                  # (Nx1) array
        grad2 <- t(hidden) %*% delH                       # ((M+1)x1) array
        delHidden <- (delH %*% t(w2)) * (1 - hidden^2)
        # (Nx(M+1)) array, and
        # tanh'(s) = 1 - (tanh(s))^2
        # delHidden = delHidden[, 2:(M+1)]
        grad1 <- t(X) %*% delHidden[, 2:(M+1)]            # ((d+1)xM) array
        # Step 3: gradient descent
        w1 <- w1 - eta * grad1
        w2 <- w2 - eta * grad2
        
    }
    proc.time() - ptm  # 10.017 seconds
    
    hidden <- tanh(X %*% w1)
    hidden <- cbind(rep(1, nrow(hidden)), hidden)         # (Nx(M+1)) array
    H <- tanh(hidden %*% w2)                              # hypothesis (Nx1) array
    ypred <- sign(H)
    ypred[ypred == 0] <- 1
    Ein <- mean(as.numeric(ypred != y))
    
}



## Part 2: calculate avg Eout with training parameters
if (part == 2) {
    # Training parameters
    M <- 3                                            # d-M-1 ANN, M in c(1,6,11,16,21)
    r <- .1
    eta <- .01                                        # d-M-1 ANN, eta in c(.001,.01,.1,1,10)
    # Eout avg = 0.0759 for M = 3, r = .1, eta = .1
    # Eout avg = 0.0365 for M = 3, r = .1, eta = .01
    # Eout avg = 0.0360 for M = 3, r = .1, eta = .001
    
    # Backprop algorithm
    Eout_arr <- c()
    ptm <- proc.time()
    for (exp in 1:500) {
        
        # weights randomly (uniformly) initialized
        w1 <- runif((d+1)*M, -r, r)
        w1 <- matrix(w1, (d+1), M)                        # ((d+1)xM) array
        w2 <- runif((M+1)*1, -r, r)
        w2 <- matrix(w2, (M+1), 1)                        # ((M+1)x1) array
        for (iter in 1:5e4) {
            # Step 1: feedforward
            hidden <- tanh(X %*% w1)
            hidden <- cbind(rep(1, N), hidden)                # (Nx(M+1)) array
            H <- tanh(hidden %*% w2)                          # hypothesis (Nx1) array
            # Step 2: backprop and get the gradient of SQUARED ERROR
            delH <- -2 * (y - H) * (1 - H^2)                  # (Nx1) array
            grad2 <- t(hidden) %*% delH                       # ((M+1)x1) array
            delHidden <- (delH %*% t(w2)) * (1 - hidden^2)
            # (Nx(M+1)) array, and
            # tanh'(s) = 1 - (tanh(s))^2
            # delHidden = delHidden[, 2:(M+1)]
            grad1 <- t(X) %*% delHidden[, 2:(M+1)]            # ((d+1)xM) array
            # Step 3: gradient descent
            w1 <- w1 - eta * grad1
            w2 <- w2 - eta * grad2
        }
        
        # calculate Eout from the test set
        hidden <- tanh(Xtest %*% w1)
        hidden <- cbind(rep(1, nrow(hidden)), hidden)         # (Nx(M+1)) array
        H <- tanh(hidden %*% w2)                              # hypothesis (Nx1) array
        ypred <- sign(H)
        ypred[ypred == 0] <- 1
        Eout <- mean(as.numeric(ypred != ytest))
        Eout_arr <- c(Eout_arr, Eout)
    }
    proc.time() - ptm  # > 814.321 seconds
    cat('Averaged Eout:', mean(Eout_arr))  # avg Eout = 0.0364
}



## Part 3: deepen the ANN architecture
if (part == 3) {
    # Training parameters
    M1 <- 8                                           # d-M1-M2-1 ANN
    M2 <- 3
    r <- .1
    eta <- .01                                        # d-M-1 ANN, eta in c(.001,.01,.1,1,10)
    
    # Backprop algorithm
    Eout_arr <- c()
    ptm <- proc.time()
    for (exp in 1:50) {
        
        # weights randomly (uniformly) initialized
        w1 <- runif((d+1)*M1, -r, r)
        w1 <- matrix(w1, (d+1), M1)                         # ((d+1)xM1) array
        w2 <- runif((M1+1)*M2, -r, r)
        w2 <- matrix(w2, (M1+1), M2)                        # ((M1+1)xM2) array
        w3 <- runif((M2+1)*1, -r, r)
        w3 <- matrix(w3, (M2+1), 1)                         # ((M2+1)x1) array
        for (iter in 1:5e4) {
            
            # Step 1: feedforward
            hidden1 <- tanh(X %*% w1)
            hidden1 <- cbind(rep(1, N), hidden1)                  # (Nx(M1+1)) array
            hidden2 <- tanh(hidden1 %*% w2)
            hidden2 <- cbind(rep(1, N), hidden2)                  # (Nx(M2+1)) array
            H <- tanh(hidden2 %*% w3)                             # hypothesis (Nx1) array
            
            # Step 2: backprop and get the gradient of SQUARED ERROR
            delH <- -2 * (y - H) * (1 - H^2)                      # (Nx1) array
            delHidden2 <- (delH %*% t(w3)) * (1 - hidden2^2)      # (Nx(M2+1)) array
            delHidden1 <- (delHidden2[, 2:(M2+1)] %*% t(w2)) * (1 - hidden1^2)    # (Nx(M1+1)) array
            grad3 <- t(hidden2) %*% delH                          # ((M2+1)x1) array
            grad2 <- t(hidden1) %*% delHidden2[, 2:(M2+1)]        # ((M1+1)xM2) array
            grad1 <- t(X) %*% delHidden1[, 2:(M1+1)]              # ((d+1)xM1) array
            
            # Step 3: gradient descent
            w1 <- w1 - eta * grad1
            w2 <- w2 - eta * grad2
            w3 <- w3 - eta * grad3
        }
        
        # calculate Eout from the test set
        hidden1 <- tanh(Xtest %*% w1)
        hidden1 <- cbind(rep(1, nrow(hidden1)), hidden1)          # (Nx(M1+1)) array
        hidden2 <- tanh(hidden1 %*% w2)
        hidden2 <- cbind(rep(1, nrow(hidden2)), hidden2)          # (Nx(M2+1)) array
        H <- tanh(hidden2 %*% w3)                                 # hypothesis (Nx1) array
        ypred <- sign(H)
        ypred[ypred == 0] <- 1
        Eout <- mean(as.numeric(ypred != ytest))
        Eout_arr <- c(Eout_arr, Eout)
    }
    proc.time() - ptm  # > 154.484 seconds
    cat('Averaged Eout:', mean(Eout_arr))  # avg Eout = 0.03704
}




