rm(list = ls())

## Import dataset and setting
dat <- read.table("hw1_15_train.dat")
X <- dat[, 1:ncol(dat)-1]
X <- as.matrix(X)
X <- cbind(rep(1, nrow(X)), X)    # (mxn) array training examples
y <- dat[[ncol(dat)]]
w <- rep(0, ncol(X))              # (nx1) array initial weights
i <- 1
update <- 0
numMistake <- 0

## Running PLA with a naive cycle
repeat {
  # Reset order i and numMistake to rerun a naive cycle
  if (i == nrow(X)) {
    i <- 1
    numMistake <- 0
  }
  
  # PLA
  h <- X[i, ]%*%w
  if (h > 0) {
    h <- 1
  } else {
    h <- -1
  }
  
  if (h != y[i]) {
    w <- w + X[i, ] * y[i]
    update <- update + 1
    numMistake <- numMistake + 1
  }
  
  i <- i + 1
  # Break condition: as with no error in the whole cycle
  if (i == nrow(X) & numMistake == 0) {
    break
  }
}


## Calculate PLA with pre-determined random cycles with random seed for each
## Also calculate the average number of updates
updateArr <- c()
for (i in 1:2000) {
  s <- sample(nrow(X), nrow(X))
  
  ## Use random seed for running PLA
  w <- rep(0, ncol(X))
  k <- 1
  update <- 0
  numMistake <- 0
  repeat {
    # reset order k and numMistake to rerun a naive cycle
    if (k == nrow(X)) {
      k <- 1
      numMistake <- 0
    }
    
    # PLA
    h <- X[s[k], ]%*%w
    if (h > 0) {
      h <- 1
    } else {
      h <- -1
    }
    
    etta <- 0.5
    if (h != y[s[k]]) {
      w <- w + X[s[k], ] * y[s[k]] * etta
      update <- update + 1
      numMistake <- numMistake + 1
    }
    
    k <- k + 1
    # break condition
    if (k == nrow(X) & numMistake == 0) {
      break
    }
  }
  # Save each number of update
  updateArr <- c(updateArr, update)
  
}
mean(updateArr)


