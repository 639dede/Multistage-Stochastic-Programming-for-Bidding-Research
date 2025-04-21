# Copyright 2022, Gurobi Optimization, LLC
#
# This example reads a MIP model from a file, adds artificial
# variables to each constraint, and then minimizes the sum of the
# artificial variables.  A solution with objective zero corresponds
# to a feasible solution to the input model.
# We can also use FeasRelax feature to do it. In this example, we
# use minrelax=1, i.e. optimizing the returned model finds a solution
# that minimizes the original objective, but only from among those
# solutions that minimize the sum of the artificial variables.

library(Matrix)
library(gurobi)

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) {
  stop('Usage: Rscript feasopt.R filename\n')
}

# Set up parameters
params <- list()
params$logfile <- 'feasopt.log'

# Read model
cat('Reading model',args[1],'...')
model <- gurobi_read(args[1], params)
cat('... done\n')

# Create penalties
penalties     <- list()
penalties$lb  <- Inf
penalties$ub  <- Inf
penalties$rhs <- rep(1,length(model$rhs))

result <- gurobi_feasrelax(model, 0, TRUE, penalties, params = params)

# Display results
if (result$feasobj > 1e-6) {
  cat('Model',args[1],'is infeasible within variable bounds\n')
} else {
  cat('Model',args[1],'is feasible\n')
}

# Clear space
rm(params, model, penalties, result)

