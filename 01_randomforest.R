#### 01. Random forest model
## Memedovich et al. 2025 - accepted at PLoS Med.

#### 1 Setup ####
# Load packages
library(tidyverse)
library(ranger)

## project folder location
# path <- ""
# setwd(path)

# Read in data
df <- readRDS("data\\df_processed_0708.RDS")

### Select variables and transform to unordered factor
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
    ) %>%
    # recode unordered factors
    mutate(across(c(challenged5:patent_count_cat),
                ~as.factor(.)))

##### 2. Set up data for model ######
### test-train split
set.seed(587); train <- sample(1:nrow(df), .8*nrow(df))

#train
df_train <- df[train,]
df_test <- df[-train,]

df_train <- df[train,]
df_test <- df[-train,]

### Set holdout IDs as weight
wts <- as.integer(if_else(
  (df$nme.brand %in% df_train$nme.brand), 
  1,0))

# classification model
rft <- ranger(challenged5 ~.-nme.brand,
              num.trees = 15000,
              write.forest = TRUE,
              holdout = TRUE,
              case.weights = wts,
              classification = TRUE,
              importance = "permutation",
              seed = 587, # Calgary area code
              data = df)

pred.train <- predict(rft, 
                      data = df_test, 
                      seed = 587)
confusion_rft <- table(df_test$challenged5, 
                         pred.train$predictions)
confusion_rft
### Output pasted
#         predicted
# true   0  1
# 0     12  3
# 1     5 22


### Calculate AUC ###
library(pROC) 
df_predicted <- tibble(true = as.numeric(df_test$challenged5)-1,
                       pred = as.numeric(pred.train$predictions)-1)
roc(df_predicted$true,df_predicted$pred)

# oob error (
rft$prediction.error

# confusion matrix  
rft$confusion.matrix
# > rft$confusion.matrix
#       predicted
# true  0   1  <NA>
#   0  12   3   76
#   1   5  22   92

### Brier score with probability model ###
rft_prob <- ranger(challenged5 ~.-nme.brand,
              num.trees = 15000,
              write.forest = TRUE,
              holdout = TRUE,
              case.weights = wts,
              probability = TRUE,
              importance = "permutation",
              seed = 587,
              data = df)

rft_prob$prediction.error 

# Extract variable importance metrics for the classification random forests
df_vi <- rft$variable.importance %>%
  enframe(name = "Variable", 
          value = "PVI") %>%
  arrange(desc(PVI)) %>%
  mutate(Variable = factor(Variable, 
                           levels = paste0(Variable)))

### initial plot of feature importance metric ###
df_vi %>% ggplot(aes(x = Variable, y = PVI)) +
  geom_bar(stat = "identity", position = "dodge",
           orientation = "x") +
  ylab("Permutation variable importance") +
  xlab("Variable") +
  labs(title = "Random forest variable importance (full dataset)")


