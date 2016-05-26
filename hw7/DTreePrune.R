# Training with decision tree algorithm
DTreePrune <- function (dat, treeLevel) {
    
    # Prune tree with specific treeLevel and return the purer label
    purity0 <- mean(as.numeric(dat[, ncol(dat)] == -1))^2
    purity1 <- mean(as.numeric(dat[, ncol(dat)] == 1))^2
    if (treeLevel == 2) {
        if (purity1 >= purity0) {
            g <- 1
            return(g)
        } else {
            g <- -1
            return(g)
        }
    }
    # Gini index as impurity measure
    imp <- 1 - purity0 - purity1
    if (imp == 0) {
        # return label
        g <- dat[, ncol(dat)][1]
        g
    } else {
        # b(x) bi-branch by purifying through stump decision function
        # For each feature i, run the stump and get theta
        impurity <- 2
        for (i in 1:(ncol(dat)-1)) {
            dat <- dat[order(dat[,i]), ] # sorting the data by feature i
            
            # Looking up theta to branch training examples
            for (j in 2:nrow(dat)) {
                theta_tmp <- (dat[j-1,i] + dat[j,i]) / 2
                dat1_tmp <- dat[1:(j-1),]
                purity0 <- mean(as.numeric(dat1_tmp[, ncol(dat1_tmp)] == -1))^2
                purity1 <- mean(as.numeric(dat1_tmp[, ncol(dat1_tmp)] == 1))^2
                imp1 <- 1 - purity0 - purity1
                dat2_tmp <- dat[j:nrow(dat),]
                purity0 <- mean(as.numeric(dat2_tmp[, ncol(dat2_tmp)] == -1))^2
                purity1 <- mean(as.numeric(dat2_tmp[, ncol(dat2_tmp)] == 1))^2
                imp2 <- 1 - purity0 - purity1
                # weighted impurity measure
                imp <- (nrow(dat1_tmp)/nrow(dat)) * imp1 + (nrow(dat2_tmp)/nrow(dat)) * imp2
                
                if (imp < impurity) {
                    impurity <- imp
                    theta <- theta_tmp
                    idx <- i # record which dimension determines the stump
                    dat1 <- dat1_tmp
                    dat2 <- dat2_tmp
                }
            }
        }
        rm(dat,dat1_tmp,dat2_tmp,theta_tmp,i,j,purity0,purity1,imp1,imp2)
        
        g1 <- DTreePrune(dat1, treeLevel+1)
        g2 <- DTreePrune(dat2, treeLevel+1)
        numNode <- 1
        if (length(g1[[1]]) != 1) {
            numNode <- numNode + g1[[1]][3]
        }
        if (length(g2[[1]]) != 1) {
            numNode <- numNode + g2[[1]][3]
        }
        b <- c(idx,theta,numNode)
        G <- list()
        G[[1]] <- b
        G[[2]] <- g1
        G[[3]] <- g2
        G
    }
}