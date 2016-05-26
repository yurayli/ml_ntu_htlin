## Can be used only in 2-level tree
# Representation:
#   g[1] = index, g[2] = threshold, g[3] = branch 1, g[4] = branch 2
StumpTree <- function (dat) {
    
    # Gini index as impurity measure
    # Training with stump
    impurity <- 2
    for (i in 1:(ncol(dat)-1)) {
        dat <- dat[order(dat[,i]), ] # sorting the data by feature i
        
        # Looking up theta to branch training examples
        for (j in 2:nrow(dat)) {
            theta_tmp <- (dat[j-1,i] + dat[j,i]) / 2
            dat1 <- dat[1:(j-1),]
            purity0 <- mean(as.numeric(dat1[, ncol(dat1)] == -1))^2
            purity1 <- mean(as.numeric(dat1[, ncol(dat1)] == 1))^2
            imp1 <- 1 - purity0 - purity1
            if (purity1 >= purity0) {
                g1_tmp <- 1
            } else {
                g1_tmp <- -1
            }
            dat2 <- dat[j:nrow(dat),]
            purity0 <- mean(as.numeric(dat2[, ncol(dat2)] == -1))^2
            purity1 <- mean(as.numeric(dat2[, ncol(dat2)] == 1))^2
            imp2 <- 1 - purity0 - purity1
            if (purity1 >= purity0) {
                g2_tmp <- 1
            } else {
                g2_tmp <- -1
            }
            # weighted impurity measure
            imp <- (nrow(dat1)/nrow(dat)) * imp1 + (nrow(dat2)/nrow(dat)) * imp2
            
            if (imp < impurity) {
                impurity <- imp
                theta <- theta_tmp
                idx <- i # record which dimension determines the stump
                g1 <- g1_tmp
                g2 <- g2_tmp
            }
        }
    }
    rm(dat,dat1,dat2,theta_tmp,i,j,purity0,purity1,imp1,imp2,g1_tmp,g2_tmp)
    g <- c(idx, theta, g1, g2)
    g
    
}

