## Implement AdaBoost-stump algorithm for T = 300 iterations
## final hypothesis G = sign(sum(alpha_t * g_t)), where g_t is the hypothesis in each iter

rm(list = ls())
# Import data
train <- read.table("hw6_adaboost_train.dat")
test <- read.table("hw6_adaboost_test.dat")

# Decision stump algorithm: g_s,i,theta(x) = s * sign(x_i - theta)
# For one iter, run decision stump for each feature i, select the one with smaller Ein
u <- rep(1/nrow(train), nrow(train))
s_arr <- c()
theta_arr <- c()
idx_arr <- c()
alpha_coef <- c()
minEps <- 1
for (iter in 1:300) {
    
    # Train small hypothesis in each iteration (weak learner)
    # For each feature i, run the decision stump and get (s, theta)
    Ein <- 1
    for (i in 1:(ncol(train)-1)) {
        dat <- cbind(train[, c(i,ncol(train))], u)
        dat <- dat[order(dat[,1]), ] # sorting the data by feature i
        
        # Looking up theta from training examples
        for (j in 1:nrow(dat)) {
            if (j == 1) {
                theta_tmp <- dat[1,1] - 1
            }
            else {
                theta_tmp <- (dat[j-1,1] + dat[j,1]) / 2
            }
            ypred <- +1 * sign(dat[,1] - theta_tmp)
            E <- mean(dat[,3] %*% as.numeric(ypred != dat[,2]))
            if (E < Ein) {
                Ein <- E
                theta <- theta_tmp
                s <- +1
                idx <- i # record which dimension determines the stump
            }
            ypred <- -1 * sign(dat[,1] - theta_tmp)
            E <- mean(dat[,3] %*% as.numeric(ypred != dat[,2]))
            if (E < Ein) {
                Ein <- E
                theta <- theta_tmp
                s <- -1
                idx <- i
            }
        }
    }
    ypred <- s * sign(train[,idx] - theta) # get the small hypothsis g_t
    
    # Calculating AdaBoost params for next iteration
    epsilon <- sum(u[which(ypred != train[,ncol(train)])]) / sum(u)
    if (epsilon < minEps) {
        minEps <- epsilon
    }
    scal <- sqrt((1-epsilon)/epsilon)
    u[which(ypred != train[,ncol(train)])] <- u[which(ypred != train[,ncol(train)])] * scal
    u[which(ypred == train[,ncol(train)])] <- u[which(ypred == train[,ncol(train)])] / scal
    alpha <- log(scal)
    
    # Recording params for calculating big G
    s_arr <- c(s_arr, s)
    theta_arr <- c(theta_arr, theta)
    idx_arr <- c(idx_arr, idx)
    alpha_coef <- c(alpha_coef, alpha)
    
}
rm(E, theta_tmp, s, theta, idx, alpha)

# To predict (measure Ein(G) here)
G <- 0
for (k in 1:300) {
     # calculate G
    G <- G + alpha_coef[k] * s_arr[k] * sign(train[, idx_arr[k]] - theta_arr[k])
}
G <- sign(G)
EinG <- mean(as.numeric(G != train[,ncol(train)]))

# To predict (measure Eout(G) here)
G <- 0
for (k in 1:300) {
     # calculate G
    G <- G + alpha_coef[k] * s_arr[k] * sign(test[, idx_arr[k]] - theta_arr[k])
}
G <- sign(G)
EoutG <- mean(as.numeric(G != test[,ncol(test)]))

