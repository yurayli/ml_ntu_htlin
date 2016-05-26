## Optimization problems for function
##      E <- exp(u) + exp(2*v) + exp(u*v) + u^2 - 2*u*v + 2*v^2 - 3*u - 2*v
## Use Gradient descent and Newton's method

## Gradient descent
u <- 0; v <- 0; eta <- .01

for (i in 1:5) {
    delU <- exp(u) + v*exp(u*v) + 2*u - 2*v - 3        # dE/du
    delV <- 2*exp(2*v) + u*exp(u*v) - 2*u + 4*v - 2    # dE/dv
    
    tmpvec <- c(u,v) - eta * c(delU,delV)
    u <- tmpvec[1]
    v <- tmpvec[2]
    
}
E <- exp(u) + exp(2*v) + exp(u*v) + u^2 - 2*u*v + 2*v^2 - 3*u - 2*v
E

## Newton's method
u <- 0; v <- 0
for (i in 1:5) {
    delU <- exp(u) + v*exp(u*v) + 2*u - 2*v - 3        # dE/du
    delV <- 2*exp(2*v) + u*exp(u*v) - 2*u + 4*v - 2    # dE/dv
    a <- exp(u) + v^2*exp(u*v) + 2                     # d^2E/du^2
    b <- (1 + u*v) * exp(u*v) - 2                      # d^2E/dudv
    c <- 4*exp(2*v) + u^2*exp(u*v) + 4                 # d^2E/dv^2
    Hessian <- matrix(c(a,b,b,c), 2, 2)                # Hessian matrix = [a b; b c]
    
    tmpvec <- c(u,v) - solve(Hessian) %*% c(delU, delV)
    u <- tmpvec[1]
    v <- tmpvec[2]
    
}
E <- exp(u) + exp(2*v) + exp(u*v) + u^2 - 2*u*v + 2*v^2 - 3*u - 2*v
E
