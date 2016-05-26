## Processed US Postal Service Zip Code dataset with extracted
## features of intensity and symmetry for training and testing
## http://www.amlbook.com/data/zip/features.train
## http://www.amlbook.com/data/zip/features.test
##
## The format of each row is "digit intensity symmetry"
## Consider binary classification problems of the form "one of the digits"
## (as the positive class) versus "other digits" (as the negative class)

rm(list = ls())


library(e1071) # libsvm in r
test <- read.table("./hw5_test.dat")
xtest <- test[,2:3]
ytest <- rep(0, nrow(test)); ytest[test[,1] == 0] <- 1; ytest <- ytest*2 - 1
train <- read.table("./hw5_train.dat")
x <- train
y <- rep(0, nrow(x)); y[x[,1] == 0] <- 1; y <- y*2 - 1         #labels of "0", "not 0"
y2 <- rep(0, nrow(x)); y2[x[,1] == 2] <- 1; y2 <- y2*2 - 1     #labels of "2", "not 2"
y4 <- rep(0, nrow(x)); y4[x[,1] == 4] <- 1; y4 <- y4*2 - 1     #labels of "4", "not 4"
y6 <- rep(0, nrow(x)); y6[x[,1] == 6] <- 1; y6 <- y6*2 - 1     #labels of "6", "not 6"
y8 <- rep(0, nrow(x)); y8[x[,1] == 8] <- 1; y8 <- y8*2 - 1     #labels of "8", "not 8"
x <- x[,2:3]


case <- 4
## Part 1: linear kernel
if (case == 1) {
    model <- svm(x = x, y = y, type = 'C-classification', kernel = 'linear',
                 scale = F, cost = 0.01, cross = 5)
    pred <- predict(model, x)
    accu <- mean(as.numeric(as.numeric(as.character(pred)) == y))  # accu of train set
    Ein <- mean(as.numeric(as.numeric(as.character(pred)) != y))
    
    # extract model params 
    # sv_coef (alpha), w, b, corresponding support vectors (alpha > 0)
    SV <- model$SV  # resulting support vectors
    alpha_y <- model$coefs  # corresponding coefficients times the training labels
    b <- -model$rho  # negative intercept
    w <- t(alpha_y) %*% SV
    if (y[model$index][1] == -1) {
        # because alpha_y[1] is always positive
        # all sign of y is inverted if y[model$index][1] = -1
        w <- -w
        b <- -b
    }
    w_mag <- sqrt(sum(w^2)) # |w| = 0.5713
}

## Part 2: polynomial kernel
if (case == 2) {
    labels <- list(y,y2,y4,y6,y8)
    minEin <- 1
    maxSum <- 0
    idx <- 1
    for (label in labels) {
        model <- svm(x = x, y = label, type = 'C-classification', kernel = 'polynomial',
                     scale = F, cost = 0.01, coef0 = 1, gamma = 1, degree = 2, cross = 5)
        pred <- predict(model, x)
        Ein <- mean(as.numeric(as.numeric(as.character(pred)) != label))
        alpha_y <- model$coefs
        if (label[model$index][1] == -1) {
            alpha_y <- -alpha_y
        }
        sum_alpha <- sum(alpha_y * label[model$index])
        if (Ein < minEin) {
            minEin <- Ein  # minEin = 0.0743
            labelIdx <- idx  # 5 -> labels of ("8", "not 8") with min Ein
        }
        if (sum_alpha > maxSum) {
            maxSum <- sum_alpha  # 21.78
        }
        idx <- idx + 1
    }
    
}

## Part 3: rbf kernel
if (case == 3) {
    Gamma <- c(1, 10, 100, 1000, 10000)
    minEout <- 1
    idx <- 1
    for (g in Gamma) {
        model <- svm(x = x, y = y, type = 'C-classification', kernel = 'radial',
                      scale = F, cost = 0.1, gamma = g, cross = 5)
        pred <- predict(model, xtest)
        Eout <- mean(as.numeric(as.numeric(as.character(pred)) != ytest))
        if (Eout < minEout) {
            minEout <- Eout  # minEout = 0.0991
            gammaIdx <- idx  # 2 -> gamma = 10
        }
        idx <- idx + 1
    }
    
}

## Part 4: cross-validation for best params (C, gamma) (validation for 100 times, about 5 min)
if (case == 4) {
    Gamma <- c(1, 10, 100, 1000, 10000)
    bestGammas <- c()
    for (i in 1:100) {
        valIndex <- sample(nrow(x), 1000)
        xval <- x[valIndex, ]
        yval <- y[valIndex]
        xtrain <- x[-valIndex, ]
        ytrain <- y[-valIndex]
        Evals <- c()
        for (g in Gamma) {
            model <- svm(x = xtrain, y = ytrain, type = 'C-classification', kernel = 'radial',
                          scale = F, cost = 0.1, gamma = g)
            pred <- predict(model, xval)
            Eval <- mean(as.numeric(as.numeric(as.character(pred)) != yval))
            Evals <- c(Evals, Eval)
        }
        bestGamma <- Gamma[which(Evals == min(Evals))[1]]
        bestGammas <- c(bestGammas, bestGamma)
    }
    numGam <- c(length(which(bestGammas == 1)), length(which(bestGammas == 10)), length(which(bestGammas == 100)),
                length(which(bestGammas == 1e3)), length(which(bestGammas == 1e4)))
    cat('Most selected gamma:', Gamma[which(numGam == max(numGam))])  # gamma = 10
}

