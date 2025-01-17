#### 02. Random forest model
## Memedovich et al. 2025 - accepted at PLoS Med.
# glmnet code adapted from rpubs tutorial (noted below) and glmnet package documentation 

#### 1 Setup ####
# Load packages
library(tidyverse)
library(glmnet)

## project folder location
# path <- ""
# setwd(path)

# Read in data
df <- readRDS("data\\df_processed_0708.RDS")

df <- df %>%
  select(
    #name brand (ID)
    nme.brand, 
    # Outcome - challenged in first year of eligibility, 0/1
    challenged5, 
    # ATC class and route of adminsitration
    atc, route,
    # Drug entry classes (described in methods), 0/1
    first.in.class, accelerated, priority,
    fast.track, breakthrough, orphan,
    # market value decile indicator
    market_cat_deciles,
    # Number of associated patents - quartiles
    patent_count_cat
    ) 

# Separate outcome
y <- as.factor(df$challenged5)

df <- df %>%
  select(
    -nme.brand, 
    -challenged5) %>% makeX(.)


##### prepare for model ####
set.seed(587)
train <- sample(1:nrow(df), .8*nrow(df))
y_train <-y[train]
x_train <-df_predictors[train,] 
y_test <-y[-train]
x_test <-df_predictors[-train,]

### manually set replicable fold_id
five_fold_ids<- sample(rep(1:5,length.out = length(y_train)))

### cv model - ridge regression ###
cvridgefit <- cv.glmnet(x_train, y_train, family = "binomial", 
                        nfolds = 5,
                        alpha=0, # ridge
                        foldid = five_fold_ids,
                        keep = TRUE, 
                        type.measure = "mse")


ridge_performance <- assess.glmnet(cvridgefit , newx=x_test, newy = y_test, 
                                   family = "binomial",
                                   s = "lambda.min")

#### cv model - LASSO regression ###
cvlassofit <- cv.glmnet(x_train, y_train, family = "binomial", 
                        nfolds = 5,
                        alpha=1, # 1 = lasso
                        foldid = five_fold_ids,
                        keep = TRUE, 
                        type.measure = "mse")

lasso_performance <- assess.glmnet(cvlassofit , 
                                   newx=x_test, 
                                   newy = y_test, 
                                   family = "binomial",
                                   s = "lambda.min")

#### not run: compare the two
# cbind(ridge_performance, lasso_performance)


############ Elastic net code #############
# Adapted from Kelly JM (2022). Ridge, lasso, and elastic net tutorial. 
# https://rpubs.com/jmkelly91/881590

models <- list()
for (i in 0:20) {
  name <- paste0("alpha", i/20)
  
  models[[name]] <-
    cv.glmnet(x_train, y_train, 
              foldid = five_fold_ids ,
              type.measure="mse", alpha=i/20, 
              standardize = TRUE,
              family="binomial")
}

results <- data.frame()
for (i in 0:20) {
  name <- paste0("alpha", i/20)
  
  mpred <- predict(models[[name]], 
                   s=models[[name]]$lambda.min, newx=x_test)
  
  temp_assess <- assess.glmnet(mpred, 
                               newy = y_test, 
                               family = "binomial",
                               s = "lambda.min")
  
  assessment.dev <- temp_assess$deviance
  assessment.class <- temp_assess$class
  assessment.mse <- temp_assess$mse
  assessment.mae <- temp_assess$mae
  
  temp <- data.frame(alpha=i/20, 
                     assessment.dev = assessment.dev,
                     assessment.class = assessment.class,
                     assessment.mse = assessment.mse,
                     assessment.mae = assessment.mae,
                     name = name)
  results <- rbind(results, temp)
}

plot(results$alpha, results$assessment.mse)
results %>% arrange(assessment.mse) %>% head(n = 5)
# lowest mse model is alpha = 0.95 (close to LASSO)

### Train cross-validated model with alpha ###
cv.elasticnet.fit <- cv.glmnet(x_train, y_train, family = "binomial", 
                        nfolds = 5,
                        alpha=0.95, # based on training
                        foldid = five_fold_ids,
                        standardize = TRUE,
                        keep = TRUE, 
                        type.measure = "mse")
plot(cv.elasticnet.fit)

# assess cv glmnet performance with test data 
elasticnet_performance <- assess.glmnet(cv.elasticnet.fit , 
                                   newx=x_test, 
                                   newy = y_test, 
                                   family = "binomial",
                                   s = "lambda.min")

# Get AUC and ROC plot
 elnauc <- cv.glmnet(x_train, y_train, family = "binomial", 
                        nfolds = 5,
                        alpha=0.95, # custom
                        foldid = five_fold_ids,
                    standardize = TRUE,
                        keep = TRUE, 
                        type.measure = "class")
rocs <- roc.glmnet(elnauc$fit.preval, 
                   newy = y_train)

best <- elnauc$index["min",]
plot(rocs[[best]], type = "l")
invisible(sapply(rocs, lines, col="grey"))
lines(rocs[[best]], lwd = 2,col = "red")

### Create confusion matrices ###

# Elastic net
conf_glmnet <- confusion.glmnet(cv.elasticnet.fit, 
                        newx = x_test, 
                        newy = y_test)
conf_glmnet 
# True
# Predicted  0  1 Total
# 0      6  2     8
# 1      9 25    34
# Total 15 27    42


# ridge, for comparison
cnf_ridge <- confusion.glmnet(cvridgefit, 
                        newx = x_test, 
                        newy = y_test)
cnf_ridge
# True
# Predicted  0  1 Total
# 0      9  5    14
# 1      6 22    28
# Total 15 27    42

# LASSO, for comparison
cnf_lasso <- confusion.glmnet(cvlassofit, 
                              newx = x_test, 
                              newy = y_test)
cnf_lasso
# True
# Predicted  0  1 Total
# 0      6  2     8
# 1      9 25    34
# Total 15 27    42



