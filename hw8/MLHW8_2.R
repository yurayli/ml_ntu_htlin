## Implement of k-Nearst Neighbor
## For classification problem

rm(list = ls())

## Import dataset
train <- read.table("hw4_knn_train.dat")
test <- read.table("hw4_knn_test.dat")

## Training parameters
N <- nrow(train)                                  # num of examples
d <- ncol(train)-1                                # num of features
X <- train[, 1:d]; y <- train[, d+1]
X <- as.matrix(X); y <- as.numeric(y)
Xtest <- test[, 1:d]; ytest <- test[, d+1]
Xtest <- as.matrix(Xtest); ytest <- as.numeric(ytest)
k <- 5

## kNN (closest |x-xn|) calculate Ein
ypred <- c()
for (num_eval in 1:N) {
    # choose the former k examples as "temporary" nrst ones
    nrst <- c()
    for (i in 1:k) {
        nrst <- c(nrst, (X[num_eval, ]-X[i, ]) %*% (X[num_eval, ]-X[i, ]))
    }
    nrst_label <- y[1:k]
    # look up other examples for real nrst ones
    for (i in (k+1):N) {
        dist <- (X[num_eval, ]-X[i, ]) %*% (X[num_eval, ]-X[i, ])
        if (dist < max(nrst)) {
            frst <- which(nrst == max(nrst))
            nrst[frst[1]] <- dist
            nrst_label[frst[1]] <- y[i]
        }
    }
    ypred <- c(ypred, sign(sum(nrst_label)))
}
Ein <- mean(as.numeric(ypred != y)) # 0.16

## kNN calculate Eout
ypred <- c()
for (num_eval in 1:nrow(Xtest)) {
    # choose the former k examples as "temporary" nrst ones
    nrst <- c()
    for (i in 1:k) {
        nrst <- c(nrst, (Xtest[num_eval, ]-X[i, ]) %*% (Xtest[num_eval, ]-X[i, ]))
    }
    nrst_label <- y[1:k]
    # look up other examples for real nrst ones
    for (i in (k+1):N) {
        dist <- (Xtest[num_eval, ]-X[i, ]) %*% (Xtest[num_eval, ]-X[i, ])
        if (dist <= max(nrst)) {
            frst <- which(nrst == max(nrst))
            nrst[frst[1]] <- dist
            nrst_label[frst[1]] <- y[i]
        }
    }
    ypred <- c(ypred, sign(sum(nrst_label)))
}
Eout <- mean(as.numeric(ypred != ytest)) # 0.316


