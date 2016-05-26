rm(list = ls())
# Import data
train <- read.table("hw7_train.dat")
test <- read.table("hw7_test.dat")
source('./DTree.R')


part <- 2.2
## Part 1: implement the simple C&RT Decision Tree algorithm without pruning 
## using the Gini index as the impurity measure. 

if (part == 1) { # Self-made algorithm
    Gt <- DTree(train)  # return trained clf
    numNode <- Gt[[1]][3]  # number of internal nodes (branching functions)
    # Data evaluation
    dat <- test  # or other data
    ypred <- c()
    Geval <- Gt
    for (i in 1:nrow(dat)) {
        x = as.vector(as.matrix(dat[i,1:2]))
        repeat { # evaluate the label of each input example
            if (length(Geval[[1]]) == 1) {
                label <- Geval[[1]]
                break
            } else {
                if (x[Geval[[1]][1]] < Geval[[1]][2]) {
                    Geval <- Geval[[2]]  # one subtree
                } else {
                    Geval <- Geval[[3]]  # the other subtree
                }
            }
        }
        ypred <- c(ypred,label)
        Geval = Gt
    }
    Eout <- 1 - mean(as.numeric(ypred == dat[,3])) # Eout = 0.126, Ein = 0

}

if (part == 1.1) { # Use built-in packages 'rpart'
    library(rpart)
    # x <- subset(train, select = -V3)
    clf <- rpart(V3 ~ ., data = train)
    plot(clf); text(clf)
    pred <- predict(clf, train[,-3])
    
    pred[pred >= 0] <- 1; pred[pred < 0] <- -1
    Ein <- 1 - mean(as.numeric(pred == train[,3])) # Ein = 0.07
    pred <- predict(clf, test[,-3])
    pred[pred >= 0] <- 1; pred[pred < 0] <- -1
    Eout <- 1 - mean(as.numeric(pred == test[,3])) # Eout = 0.135
    
}



## -----------------------------------------------------------------------
## Part 2: implement the Random Forest (sampled data size N' = N per tree)
## with 300 trees for 100 times experiment

if (part == 2) { # Self-made algorithm (very very slow)
    ptm <- proc.time()
    N <- nrow(train)
    Ein_gt <- 0
    Ein <- 0
    Eout <- 0
    for (numOfTime in 1:100) {
        
        Grf_in <- 0  # each vote for train set from each tree
        Grf_out <- 0  # each vote for test set from each tree
        
        for (t in 1:300) {
            seed <- sample(N, N, replace = T)
            trainDat <- train[seed, ]
            gt <- DTree(trainDat)  # each tree gt
            
            # evaluate train set from each tree
            ypred <- c()
            g_eval <- gt
            for (i in 1:N) {
                x = as.vector(as.matrix(train[i,1:2]))
                repeat { # evaluate the label of each input example
                    if (length(g_eval[[1]]) == 1) {
                        label <- g_eval[[1]]
                        break
                    } else {
                        if (x[g_eval[[1]][1]] < g_eval[[1]][2]) {
                            g_eval <- g_eval[[2]]  # one subtree
                        } else {
                            g_eval <- g_eval[[3]]  # the other subtree
                        }
                    }
                }
                ypred <- c(ypred,label)
                g_eval <- gt
            }
            Ein_gt <- Ein_gt + ( 1 - mean(as.numeric(ypred == train[,3])) )  # sum of Ein of each gt in each round
            Grf_in <- Grf_in + ypred  # sum of each vote from each tree
            
            # evaluate test set from each tree
            ypred <- c()
            g_eval <- gt
            for (i in 1:nrow(test)) {
                x = as.vector(as.matrix(test[i,1:2]))
                repeat {
                    if (length(g_eval[[1]]) == 1) {
                        label <- g_eval[[1]]
                        break
                    } else {
                        if (x[g_eval[[1]][1]] < g_eval[[1]][2]) {
                            g_eval <- g_eval[[2]]
                        } else {
                            g_eval <- g_eval[[3]]
                        }
                    }
                }
                ypred <- c(ypred,label)
                g_eval = gt
            }
            Grf_out <- Grf_out + ypred  # sum of each vote from each tree
            
        }
        
        # evaluate train set from G_rf
        Grf_in <- sign(Grf_in)
        Ein <- Ein + ( 1 - mean(as.numeric(Grf_in == train[,3])) )  # sum of Ein of each forest in each round
        
        # evaluate test set from G_rf
        Grf_out <- sign(Grf_out)
        Eout <- Eout + ( 1 - mean(as.numeric(Grf_out == test[,3])) )  # sum of Eout of each forest in each round
        
        #proc.time() - ptm
    }
    
    Ein_gt <- Ein_gt / 30000  # average of 30000 trees Ein_gt: .03 - .06
    Ein <- Ein / 100  # average of 100 forests Ein: 0 - .03
    Eout <- Eout / 100  # average of 100 forests Eout: .06 - .09
    proc.time() - ptm  # > 9182.517 sec > 153 min > too long......
    
}


if (part == 2.1) { # Self-made algorithm for only 1 round random forest
    ptm <- proc.time()
    N <- nrow(train)
    Ein_gt <- 0
    
    #for (numOfTime in 1:1) {
        
        Grf_in <- 0
        Grf_out <- 0
        set.seed(42)
        
        for (t in 1:300) {
            seed <- sample(N, N, replace = T)
            trainDat <- train[seed, ]
            gt <- DTree(trainDat)  # each tree gt
            
            # evaluate train set from each tree
            ypred <- c()
            g_eval <- gt
            for (i in 1:N) {
                x = as.vector(as.matrix(train[i,1:2]))
                repeat { # evaluate the label of each input example
                    if (length(g_eval[[1]]) == 1) {
                        label <- g_eval[[1]]
                        break
                    } else {
                        if (x[g_eval[[1]][1]] < g_eval[[1]][2]) {
                            g_eval <- g_eval[[2]]  # one subtree
                        } else {
                            g_eval <- g_eval[[3]]  # the other subtree
                        }
                    }
                }
                ypred <- c(ypred,label)
                g_eval <- gt
            }
            Ein_gt <- Ein_gt + ( 1 - mean(as.numeric(ypred == train[,3])) )  # sum of Ein of each gt in each round
            Grf_in <- Grf_in + ypred  # sum of each voting from each tree
            
            # evaluate test set from each tree
            ypred <- c()
            g_eval <- gt
            for (i in 1:nrow(test)) {
                x = as.vector(as.matrix(test[i,1:2]))
                repeat {
                    if (length(g_eval[[1]]) == 1) {
                        label <- g_eval[[1]]
                        break
                    } else {
                        if (x[g_eval[[1]][1]] < g_eval[[1]][2]) {
                            g_eval <- g_eval[[2]]
                        } else {
                            g_eval <- g_eval[[3]]
                        }
                    }
                }
                ypred <- c(ypred,label)
                g_eval = gt
            }
            Grf_out <- Grf_out + ypred  # sum of each voting from each tree
            
        }
        
        # evaluate train set from G_rf
        Grf_in <- sign(Grf_in)
        Ein <- ( 1 - mean(as.numeric(Grf_in == train[,3])) )  # sum of Ein of each forest in each round
                                                              # Ein = 0 for one try
        # evaluate test set from G_rf
        Grf_out <- sign(Grf_out)
        Eout <- ( 1 - mean(as.numeric(Grf_out == test[,3])) )  # sum of Eout of each forest in each round
                                                               # Eout = .073 for one try
        #proc.time() - ptm
    #}
    
    Ein_gt <- Ein_gt / 300  # average of 300 trees Ein_gt: .0512 for one try
    
    proc.time() - ptm  # > 88.91 sec
    
}

if (part == 2.2) { # Use built-in packages 'randomForest'!
    library(randomForest)
    ptm <- proc.time()
    
    set.seed(420)
    Ein <- 0
    Eout <- 0
    for (numOfTime in 1:100) {
        clf <- randomForest(V3 ~ ., data = train, ntree=300)
        pred <- predict(clf, train[,-3])
        pred[pred >= 0] <- 1; pred[pred < 0] <- -1
        Ein <- Ein + (1 - mean(as.numeric(pred == train[,3])))
        pred <- predict(clf, test)
        pred[pred >= 0] <- 1; pred[pred < 0] <- -1
        Eout <- Eout + (1 - mean(as.numeric(pred == test[,3])))
    }
    Ein <- Ein / 100  # avg Ein = .0046
    Eout <- Eout / 100  # avg Eout = .072
    
    proc.time() - ptm  # > 4.84 sec
}

if (part == 2.3) { # Use built-in packages 'randomForest' for only 1 round
    library(randomForest)
    ptm <- proc.time()
    
    set.seed(420)
    clf <- randomForest(V3 ~ ., data = train, ntree=300)
    print(clf)
    pred_train <- predict(clf, train[,-3])
    pred_train[pred_train >= 0] <- 1; pred_train[pred_train < 0] <- -1
    Ein <- 1 - mean(as.numeric(pred_train == train[,3])) # Ein = 0.01
    pred_test <- predict(clf, test)
    pred_test[pred_test >= 0] <- 1; pred_test[pred_test < 0] <- -1
    Eout <- 1 - mean(as.numeric(pred_test == test[,3])) # Eout = 0.071
    
    proc.time() - ptm  # > .066 sec
}




## -----------------------------------------------------------------------
## Part 3-1: implement the Random Forest (sampled data size N' = N per tree)
## with 300 PRUNED trees for 100 times experiment (using whole data in evaluation)
if (part == 3.1) {
    source('~/Desktop/StumpTree.R')
    ptm <- proc.time()
    N <- nrow(train)
    Ein <- 0
    Eout <- 0
    for (numOfTime in 1:100) {
        
        Grf_in <- 0
        Grf_out <- 0
        
        for (t in 1:300) {
            seed <- sample(N, N, replace = T)
            trainDat <- train[seed, ]
            gt <- StumpTree(trainDat)
            # Representation:
            #   gt[1] = index, gt[2] = threshold, gt[3] = branch 1, gt[4] = branch 2
            # evaluate train set from each tree
            ypred <- rep(0, N)
            ypred[which(train[, gt[1]] < gt[2])] <- gt[3]
            ypred[which(train[, gt[1]] >= gt[2])] <- gt[4]
            Grf_in <- Grf_in + ypred
            
            # evaluate test set from each tree
            ypred <- rep(0, nrow(test))
            ypred[which(test[, gt[1]] < gt[2])] <- gt[3]
            ypred[which(test[, gt[1]] >= gt[2])] <- gt[4]
            Grf_out <- Grf_out + ypred
            
        }
        
        # evaluate training set from G_rf
        Grf_in <- sign(Grf_in)
        Ein <- Ein + ( 1 - mean(as.numeric(Grf_in == train[,3])) )
        
        # evaluate test set from G_rf
        Grf_out <- sign(Grf_out)
        Eout <- Eout + ( 1 - mean(as.numeric(Grf_out == test[,3])) )
        
    }
    
    Ein <- Ein / 100  # average of 100 forests Ein
    Eout <- Eout / 100  # average of 100 forests Eout
    proc.time() - ptm  # > 1581.772 sec
    
}



## -----------------------------------------------------------------------
## Part 3-2: implement the Random Forest (sampled data size N' = N per tree)
## with 300 PRUNED trees for 100 times experiment  (using point-wise in evaluation)
if (part == 3.2) {
    source('./DTreePrune.R')
    ptm <- proc.time()
    N <- nrow(train)
    Ein <- 0
    Eout <- 0
    for (numOfTime in 1:100) {
        
        Grf_in <- 0
        Grf_out <- 0
        
        for (t in 1:300) {
            seed <- sample(N, N, replace = T)
            trainDat <- train[seed, ]
            gt <- DTreePrune(trainDat,1)
            
            # evaluate train set from each tree
            ypred <- c()
            g_eval <- gt
            for (i in 1:N) {
                x = as.vector(as.matrix(train[i,1:2]))
                
                if (length(g_eval[[1]]) == 1) {
                    label <- g_eval[[1]]
                } else {
                    if (x[g_eval[[1]][1]] < g_eval[[1]][2]) {
                        label <- g_eval[[2]]
                    } else {
                        label <- g_eval[[3]]
                    }
                }
                
                ypred <- c(ypred,label)
                g_eval = gt
            }
            Grf_in <- Grf_in + ypred
            
            # evaluate test set from each tree
            ypred <- c()
            g_eval <- gt
            for (i in 1:nrow(test)) {
                x = as.vector(as.matrix(test[i,1:2]))
                repeat {
                    if (length(g_eval[[1]]) == 1) {
                        label <- g_eval[[1]]
                        break
                    } else {
                        if (x[g_eval[[1]][1]] < g_eval[[1]][2]) {
                            g_eval <- g_eval[[2]]
                        } else {
                            g_eval <- g_eval[[3]]
                        }
                    }
                }
                ypred <- c(ypred,label)
                g_eval = gt
            }
            Grf_out <- Grf_out + ypred
            
        }
        
        # evaluate train set from G_rf
        Grf_in <- sign(Grf_in)
        Ein <- Ein + ( 1 - mean(as.numeric(Grf_in == train[,3])) )
        
        # evaluate test set from G_rf
        Grf_out <- sign(Grf_out)
        Eout <- Eout + ( 1 - mean(as.numeric(Grf_out == test[,3])) )
        
    }
    
    Ein <- Ein / 100  # average of 100 forests Ein
    Eout <- Eout / 100  # average of 100 forests Eout
    proc.time() - ptm  # > 5368.427 sec
    
}



