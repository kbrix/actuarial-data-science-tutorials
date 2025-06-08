library(magrittr)

dir <- dirname(parent.frame(2)$ofile) # run as source

dt <- data.table::fread(
  input = glue::glue("{dir}/embeddings.csv"),
  showProgress = TRUE)

pq <- data.table::fread(
  glue::glue("{dir}/parquet.csv"),
  showProgress = TRUE)

str(dt); str(pq)

dt[, NUMTOTV := pq$NUMTOTV]

set.seed(69)
sample <- sample.int(n = nrow(dt), size = floor(0.80 * nrow(dt)), replace = F)
train <- dt[sample, ]
test  <- dt[-sample, ]

# Numeric response and feature matrix
y <- as.factor(train$NUMTOTV)
X <- data.matrix(train[, 1:768])

# Train
model <- lightgbm::lightgbm(
  data = X, label = y,
  params = list(
    objective = "multiclass",
    num_leaves = 31,
    learning_rate = 0.05,
    min_data_in_leaf = 50, 
    min_sum_hessian_in_leaf = 0.001,
    lambda_l1 = 3,
    lambda_l2 = 5
  ), 
  nrounds = 500L,
  verbose = -1L
)

# Result
prediction <- stats::predict(model, newdata = data.matrix(test[, 1:768]))
predictedLabels <- apply(prediction, MARGIN = 1, FUN = which.max)

res <- cbind(true = test$NUMTOTV, predicted = predictedLabels) %>% data.frame()

acc <- ifelse(res$true == res$predicted, 1, 0) %>% { sum(.) / length(.) }

acc # ~88.4%, it is a bit crappy... but this script took seconds to write lol...
