## Implement of k-Means algorithm
## For clustering problem

rm(list = ls())

## Import dataset
train <- read.table("hw4_kmeans_train.dat")

## Training parameters
N <- nrow(train)                                  # num of examples
d <- ncol(train)                                  # num of features
X <- as.matrix(train)                             # no labels
k <- 2

## k-means experiments
Ein_arr <- c()
ptm <- proc.time()
for (exp in 1:500) {
    mu <- X[sample(N,k), ] # randomly initialize cluster centroids
    for (iter in 1:20) {
        # Find the closest centroid
        group <- c()
        for (i in 1:N) {
            # idx <- 0
            min <- 0
            for (j in 1:k) {
                dis <- (mu[j, ] - X[i, ]) %*% (mu[j, ] - X[i, ])
                if ((j == 1) | (dis < min)) {
                    min = dis
                    idx <- j
                }
            }
            group <- c(group, idx)
            
        }
        # Compute new centroids
        for (i in 1:k) {
            mu[i, ] <- apply(as.matrix(X[which(group == i), ]), 2, mean)
        }
    }
    
    # calculate Ein = (1/N) * sum_k(sum_i( |x_i - mu_k|^2 ))
    s <- 0
    for (i in 1:k) {
        Xi <- X[which(group == i), ]
        for (j in 1:nrow(Xi)) {
            s <- s + (Xi[j, ] - mu[i, ]) %*% (Xi[j, ] - mu[i, ])
        }
    }
    Ein <- s / N
    Ein_arr <- c(Ein_arr, Ein)
}
proc.time() - ptm 
# 20.4 seconds for k = 2, 88.7 seconds for k = 10
# avg of Ein_arr is about 2.67 for k = 2, 1.66 for k = 10



