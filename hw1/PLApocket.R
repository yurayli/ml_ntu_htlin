rm(list=ls())

## Import dataset
datTrain <- read.table("hw1_18_train.dat")
datTest <- read.table("hw1_18_test.dat")
X <- datTrain[, 1:ncol(datTrain)-1]
X <- as.matrix(X)
X <- cbind(rep(1, nrow(X)), X)                # (mxn) array training examples
y <- datTrain[[ncol(datTrain)]]
Xtest <- datTest[, 1:ncol(datTest)-1]
Xtest <- as.matrix(Xtest)
Xtest <- cbind(rep(1, nrow(Xtest)), Xtest)    # (mxnt) array test examples
ytest <- datTest[[ncol(datTest)]]
w <- rep(0, ncol(X))                          # (nx1) array initial weights
wPLA <- w
wPocket <- w

## PLA and (Modified PLA) Pocket algorithm with pre-determined random cycle
## Run 2000 times with termination of a given update number
errRatetestArr <- c()
errRateArr <- c()
for (i in 1:2000) {
  updatePLA <- 0
  updatePocket <- 0
  errRate <- 1
  
  ## Use random seed for running PLA and Pocket
  repeat {
    # rerun visiting the examples randomly until getting the update number
    k <- sample(nrow(X), 1)
    
    # PLA
    hPLA <- X[k, ] %*% wPLA
    if (hPLA > 0) {
      hPLA <- 1
    } else {
      hPLA <- -1
    }
    etta <- 1
    if (hPLA != y[k]) {
      wPLA <- wPLA + X[k, ] * y[k] * etta
      updatePLA <- updatePLA + 1
    }
    
    # Pocket algorithm (save the w causing smaller error rate)
    h <- X[k, ] %*% w
    if (h > 0) {
      h <- 1
    } else {
      h <- -1
    }
    if (h != y[k]) {
      wPocket <- w + X[k, ] * y[k] * etta
    }
    y_predict <- X %*% wPocket
    y_predict[y_predict > 0] <- 1
    y_predict[y_predict <= 0] <- -1
    errRate_tmp <- mean(as.numeric(y != y_predict))
    if (errRate_tmp < errRate) {  # Save the params with a smaller error
      w <- wPocket
      errRate <- errRate_tmp
      errRateArr <- c(errRateArr, errRate)
      updatePocket <- updatePocket + 1
    }
    
    # Terminate by updatePocket or updatePLA depending on the question of class
    # Examine the error rate of test set
    ## Note for Pocket, it's sometimes hard to increase updatePocket
    if (updatePLA == 50) {
      y_predict <- Xtest %*% w
      y_predict[y_predict > 0] <- 1
      y_predict[y_predict <= 0] <- -1
      errRatetest <- mean(as.numeric(ytest != y_predict))
      errRatetestArr <- c(errRatetestArr, errRatetest)
      break
    }
    
  }
}