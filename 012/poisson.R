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

model_formula <- dt[, 1:768] %>% names() %>% paste(collapse = " + ") %>% { paste("NUMTOTV ~ ", .) }

set.seed(69)
sample <- sample.int(n = nrow(dt), size = floor(0.80 * nrow(dt)), replace = F)
train <- dt[sample, ]
test  <- dt[-sample, ]

model_glm <- model_formula %>% reformulate() %>% glm(family = poisson(link = "log"), data = train)
prediction <- stats::predict(model_glm, newdata = test[, 1:768], type = "response")

# Compare test count values with predicted expected mean values (for fun)...
cbind(test$NUMTOTV, prediction)

# In a proper analysis, where you compare model, you should compute and compare your loss functions...