#### load library ----
library(packrat)
# on(project = "/work")
# PackratFormat: 1.4
# PackratVersion: 0.5.0
# RVersion: 3.5.1
# Repos: CRAN=https://mran.microsoft.com/snapshot/2018-12-20
# 
# Package: packrat
# Source: CRAN
# Version: 0.5.0
# Hash: 498643e765d1442ba7b1160a1df3abf9
on()
library(tidyverse)
library(lubridate)
library(ROCR)
library(caret)
library(xgboost)
library(glmnet)
library(precrec)
library(pROC)
library(xgboostExplainer)
library(readxl)

#### preprocessing ----
data.raw = read_xlsx("side_project/pepsico/shelf-life-study-data-for-analytics-challenge-v2.xlsx")
## categorical (one-hot encoding): product type, base ingredient, process type, storage conditions, *processing agent stability index
## binary: packaging stabilizer added, preservative added, *transparent window in package
## continuous: sample age, difference from fresh, moisture, residual oxygen, hexanal, *processing agent stability index
data.tidy = data.raw %>% 
  rename(study_number = `Study Number`, sample_id = `Sample ID`,
         product_type = `Product Type`, base_ingredient = `Base Ingredient`,
         process_type = `Process Type`, sample_age_wks = `Sample Age (Weeks)`,
         diff_from_fresh = `Difference From Fresh`, storage_cond = `Storage Conditions`,
         pkg_stab_added = `Packaging Stabilizer Added`, transp_window_pkg = `Transparent Window in Package`,
         process_agent_stab_idx = `Processing Agent Stability Index`, preserv_added = `Preservative Added`,
         moisture = `Moisture (%)`, resid_oxygen = `Residual Oxygen (%)`,
         hexanal_ppm = `Hexanal (ppm)`) %>% 
  mutate(product_type = if_else(is.na(product_type), "PDT_N", paste("PDT", product_type, sep = "_")),
         base_ingredient = if_else(is.na(base_ingredient), "BI_N", paste("BI", base_ingredient, sep = "_")),
         process_type = if_else(is.na(process_type), "PCT_N", paste("PCT", process_type, sep = "_")),
         storage_cond = if_else(is.na(storage_cond), "STC_N", paste("STC", storage_cond, sep = "_")),
         product_type_val = 1, base_ingredient_val = 1, process_type_val = 1, storage_cond_val = 1,
         pkg_stab_added = if_else(is.na(pkg_stab_added), "PSA_NA", paste("PSA", pkg_stab_added, sep = "_")),
         preserv_added = if_else(is.na(preserv_added), "PRA_NA", paste("PRA", preserv_added, sep = "_")),
         transp_window_pkg = if_else(is.na(transp_window_pkg), "TWP_NA", paste("TWP", transp_window_pkg, sep = "_")),
         pkg_stab_added_val = 1, preserv_added_val = 1, transp_window_pkg_val = 1,
         ## 99 for NAs in continuous vars
         moisture_NA = if_else(is.na(moisture), 0, 1), moisture = if_else(is.na(moisture), 99, moisture),
         resid_oxygen_NA = if_else(is.na(resid_oxygen), 0, 1), resid_oxygen = if_else(is.na(resid_oxygen), 99, resid_oxygen),
         hexanal_NA = if_else(is.na(hexanal_ppm), 0, 1), hexanal_ppm = if_else(is.na(hexanal_ppm), 99, hexanal_ppm)) %>% 
  spread(product_type, product_type_val, fill = 0) %>% 
  spread(base_ingredient, base_ingredient_val, fill = 0) %>% 
  spread(process_type, process_type_val, fill = 0) %>% 
  spread(storage_cond, storage_cond_val, fill = 0) %>% 
  spread(pkg_stab_added, pkg_stab_added_val, fill = 0) %>% 
  spread(preserv_added, preserv_added_val, fill = 0) %>% 
  spread(transp_window_pkg, transp_window_pkg_val, fill = 0) %>% 
  mutate(label_cont = diff_from_fresh, label_bin = if_else(diff_from_fresh > 20, 1, 0)) %>% 
  select(-diff_from_fresh)

#### data split ----
## by study id, to avoid potential data leak
data.samples = data.tidy %>% 
  distinct(study_number, sample_id)
data.studies = data.tidy %>% 
  distinct(study_number)
set.seed(2020)
rand_ind = sample(seq_len(nrow(data.studies)))
test_ind = rand_ind[(1+0.2*(i-1)*length(rand_ind)):floor(0.2*i*length(rand_ind))]
train_studies = data.studies[-test_ind,] %>% .$study_number
test_studies = data.studies[test_ind, ] %>% .$study_number


#### Logistic Regression ----
## 80-10-10 split: search for threshold in validation set, validate on test set
train = data.tidy %>% filter(study_number %in% train_studies) 
test = data.tidy %>% filter(study_number %in% test_studies) 
train_label = train$label_bin
test_label = test$label_bin
## lasso
lasso_cv = cv.glmnet(x = as.matrix(train[,3:(ncol(train)-2)]), y = as.factor(train_label), family = "binomial", alpha = 1)
lambda_1se = lasso_cv$lambda.1se
min_lambda = lasso_cv$lambda.min
lasso_fit_lambda_1se = glmnet(x = as.matrix(train[,3:(ncol(train)-2)]), y = as.factor(train_label), family = "binomial", alpha = 1, 
             lambda=lambda_1se)
lasso_fit_minlambda = glmnet(x = as.matrix(train[,3:(ncol(train)-2)]), y = as.factor(train_label), family = "binomial", alpha = 1, 
                   lambda=min_lambda)
lasso_pred_lambda_1se = predict(lasso_fit_lambda_1se, newx = as.matrix(test[,3:(ncol(test)-2)]), s = lambda_1se, type = "response")
pr_scores_lambda_1se = evalmod(scores = lasso_pred_lambda_1se[,1], labels = test_label)
aucs_lambda_1se = precrec::auc(pr_scores_lambda_1se)$aucs
print(aucs_lambda_1se)
## [1] 0.7016317 0.5071029
lasso_pred_minlambda = predict(lasso_fit_minlambda, newx = as.matrix(test[,3:(ncol(test)-2)]), s = min_lambda, type = "response")
pr_scores_minlambda = evalmod(scores = lasso_pred_minlambda[,1], labels = test_label)
aucs_minlambda = precrec::auc(pr_scores_minlambda)$aucs
print(aucs_minlambda)
## [1] 0.7139527 0.4849267
## no real difference between lambda.min and lambda.1se; using lambda.1se for better generalizibility

## ridge
ridge_cv = cv.glmnet(x = as.matrix(train[,3:(ncol(train)-2)]), y = as.factor(train_label), family = "binomial", alpha = 0)
lambda_1se = ridge_cv$lambda.1se
min_lambda = ridge_cv$lambda.min
ridge_fit_lambda_1se = glmnet(x = as.matrix(train[,3:(ncol(train)-2)]), y = as.factor(train_label), family = "binomial", alpha = 0, 
                              lambda=lambda_1se)
ridge_fit_minlambda = glmnet(x = as.matrix(train[,3:(ncol(train)-2)]), y = as.factor(train_label), family = "binomial", alpha = 0, 
                             lambda=min_lambda)
ridge_pred_lambda_1se = predict(ridge_fit_lambda_1se, newx = as.matrix(test[,3:(ncol(test)-2)]), s = lambda_1se, type = "response")
pr_scores_lambda_1se = evalmod(scores = ridge_pred_lambda_1se[,1], labels = test_label)
aucs_lambda_1se = precrec::auc(pr_scores_lambda_1se)$aucs
print(aucs_lambda_1se)
## [1] 0.7049617 0.5429877
ridge_pred_minlambda = predict(ridge_fit_minlambda, newx = as.matrix(test[,3:(ncol(test)-2)]), s = min_lambda, type = "response")
pr_scores_minlambda = evalmod(scores = ridge_pred_minlambda[,1], labels = test_label)
aucs_minlambda = precrec::auc(pr_scores_minlambda)$aucs
print(aucs_minlambda)
## [1] 0.7146187 0.5223525
## no real difference between lasso and ridge

#### XGBoost Classification ----
## 5-fold testing
start_time = proc.time()
results.list = lapply(1:5, FUN = function(i){
  ## always using the same split for model comparisons
  set.seed(2020)
  rand_ind = sample(seq_len(nrow(data.studies)))
  test_ind = rand_ind[(1+floor(0.2*(i-1)*length(rand_ind))):floor(0.2*i*length(rand_ind))]
  train_studies = data.studies[-test_ind,] %>% .$study_number
  test_studies = data.studies[test_ind, ] %>% .$study_number
  
  #### grid search ----
  ## 80-10-10 split: search for threshold in validation set, validate on test set
  train = data.tidy %>% filter(study_number %in% train_studies) 
  test = data.tidy %>% filter(study_number %in% test_studies) 
  train_label = train$label_bin
  test_label = test$label_bin
  dtrain = xgb.DMatrix(data = as.matrix(train[,3:(ncol(train)-2)]), label = train_label)
  dtest = xgb.DMatrix(data = as.matrix(test[,3:(ncol(test)-2)]), label = test_label)
  
  #### hyperparameter tuning ----
  best_param = list()
  best_seednumber = 1234
  best_iter = 0
  best_logloss = Inf
  best_logloss_index = 0
  
  ## given it's a small dataset, it's more efficient to randomly sample parameter combinations than tuning one at a time
  for (iter in 1:1000) {
    ## reset seed to ensure that the hyperparameters are independently random from data splits
    set.seed(NULL)
    seed.number = sample.int(10000, 1)[[1]]
    set.seed(seed.number)
    param <- list(objective = "binary:logistic",
                  eval_metric = "logloss",
                  booster = sample(c('gblinear', 'gbtree'),1),
                  max_depth = sample(6:17, 1),
                  eta = runif(1, .01, .3),
                  gamma = runif(1, 0.0, 0.2), 
                  subsample = runif(1, .6, 1),
                  colsample_bytree = runif(1, .5, 1), 
                  min_child_weight = sample(1:40, 1),
                  max_delta_step = sample(1:10, 1)
    )
    cv.nround = 1000
    cv.nfold = 5

    ## ensure that the data splits in 5-fold cv is consistently the same for parameter tuning
    set.seed(2020)
    mdcv <- xgb.cv(data=dtrain, params = param, nthread=6, 
                   nfold=cv.nfold, nrounds=cv.nround,
                   verbose = F, early_stopping_rounds = 8, maximize=FALSE)
    
    min_logloss = min(mdcv$evaluation_log$test_logloss_mean)
    min_logloss_index = which.min(mdcv$evaluation_log$test_logloss_mean)
    
    if (min_logloss < best_logloss) {
      best_logloss = min_logloss
      best_logloss_index = min_logloss_index
      best_seednumber = seed.number
      best_iter = iter
      best_param = param
    }
  }
  
  nround = best_logloss_index
  md = xgb.train(data=dtrain, params=best_param, nrounds=nround, nthread=6)
  
  test_pred_studies = predict(md, newdata = dtest)
  pr_scores = evalmod(scores = test_pred_studies, labels = test_label)
  aucs = precrec::auc(pr_scores)$aucs
  
  test.df = test %>% 
    mutate(test_pred_studies = test_pred_studies,
           auroc = aucs[1], auprc = aucs[2],
           best_logloss = best_logloss)
  return(test.df)
})
(proc.time() - start_time)
# user   system  elapsed 
# 3324.702  146.939  647.510 

results = bind_rows(results.list)
mean(results$auroc)
## [1] 0.7810958
mean(results$auprc)
## [1] 0.4431323
## better than lasso and ridge

# feat_imp = xgb.importance(colnames(data.tidy)[3:(ncol(data.tidy)-2)], md)
# xgb.plot.importance(feat_imp[1:20,])
## sample age appears to be the most important feature

#### output file ----
## classification
## data.raw %>% distinct(`Sample ID`, `Sample Age (Weeks)`, `Difference From Fresh`)
## 1 to 1 match, using this combination for joining the results to the original data table
data.results = data.raw %>% 
  inner_join(results %>% 
               select(sample_id, sample_age_wks, label_cont, test_pred_studies),
             by = c(`Sample ID` = "sample_id", `Sample Age (Weeks)` = "sample_age_wks", `Difference From Fresh` = "label_cont")) %>% 
  rename(Prediction = test_pred_studies)

## writing to csv first
write_csv(data.results, "/work/side_project/pepsico/JiChen.csv")

#### Linear Regression ----
## 80-10-10 split: search for threshold in validation set, validate on test set
train = data.tidy %>% filter(study_number %in% train_studies) 
test = data.tidy %>% filter(study_number %in% test_studies) 
train_label = train$label_cont
test_label = test$label_cont

## lasso
lasso_cv = cv.glmnet(x = as.matrix(train[,3:(ncol(train)-2)]), y = train_label, family = "gaussian", alpha = 1)
lambda_1se = lasso_cv$lambda.1se
min_lambda = lasso_cv$lambda.min
lasso_fit_lambda_1se = glmnet(x = as.matrix(train[,3:(ncol(train)-2)]), y = (train_label), family = "gaussian", alpha = 1, 
                              lambda=lambda_1se)
lasso_fit_minlambda = glmnet(x = as.matrix(train[,3:(ncol(train)-2)]), y = (train_label), family = "gaussian", alpha = 1, 
                             lambda=min_lambda)
lasso_pred_lambda_1se = predict(lasso_fit_lambda_1se, newx = as.matrix(test[,3:(ncol(test)-2)]), s = lambda_1se, type = "response")
lasso_1se_rmse = RMSE(lasso_pred_lambda_1se[,1], test_label)
lasso_1se_rsquared = cor(lasso_pred_lambda_1se[,1], test_label)^2

lasso_pred_minlambda = predict(lasso_fit_minlambda, newx = as.matrix(test[,3:(ncol(test)-2)]), s = min_lambda, type = "response")
lasso_minlambda_rmse = RMSE(lasso_pred_minlambda[,1], test_label)
lasso_minlambda_rsquared = cor(lasso_pred_minlambda[,1], test_label)^2
## no real difference between lambda.min and lambda.1se;

## ridge
ridge_cv = cv.glmnet(x = as.matrix(train[,3:(ncol(train)-2)]), y = train_label, family = "gaussian", alpha = 0)
lambda_1se = ridge_cv$lambda.1se
min_lambda = ridge_cv$lambda.min
ridge_fit_lambda_1se = glmnet(x = as.matrix(train[,3:(ncol(train)-2)]), y = (train_label), family = "gaussian", alpha = 0, 
                              lambda=lambda_1se)
ridge_fit_minlambda = glmnet(x = as.matrix(train[,3:(ncol(train)-2)]), y = (train_label), family = "gaussian", alpha = 0, 
                             lambda=min_lambda)
ridge_pred_lambda_1se = predict(ridge_fit_lambda_1se, newx = as.matrix(test[,3:(ncol(test)-2)]), s = lambda_1se, type = "response")
ridge_1se_rmse = RMSE(ridge_pred_lambda_1se[,1], test_label)
ridge_1se_rsquared = cor(ridge_pred_lambda_1se[,1], test_label)^2

ridge_pred_minlambda = predict(ridge_fit_minlambda, newx = as.matrix(test[,3:(ncol(test)-2)]), s = min_lambda, type = "response")
ridge_minlambda_rmse = RMSE(ridge_pred_minlambda[,1], test_label)
ridge_minlambda_rsquared = cor(ridge_pred_minlambda[,1], test_label)^2
## no real difference between lambda.min and lambda.1se;
## no real difference between lasso and ridge

#### XGBoost Regression ----
start_time = proc.time()
results.list = lapply(1:5, FUN = function(i){
  ## always using the same split for model comparisons
  set.seed(2020)
  rand_ind = sample(seq_len(nrow(data.studies)))
  test_ind = rand_ind[(1+floor(0.2*(i-1)*length(rand_ind))):floor(0.2*i*length(rand_ind))]
  train_studies = data.studies[-test_ind,] %>% .$study_number
  test_studies = data.studies[test_ind, ] %>% .$study_number
  
  #### grid search ----
  ## 80-10-10 split: search for threshold in validation set, validate on test set
  train = data.tidy %>% filter(study_number %in% train_studies) 
  test = data.tidy %>% filter(study_number %in% test_studies) 
  train_label = train$label_cont
  test_label = test$label_cont
  dtrain = xgb.DMatrix(data = as.matrix(train[,3:(ncol(train)-2)]), label = train_label)
  dtest = xgb.DMatrix(data = as.matrix(test[,3:(ncol(test)-2)]), label = test_label)
  
  #### hyperparameter tuning ----
  best_param = list()
  best_seednumber = 1234
  best_iter = 0
  best_rmse = Inf
  best_rmse_index = 0
  
  ## given it's a small dataset, it's more efficient to randomly sample parameter combinations than tuning one at a time
  for (iter in 1:10) {
    ## reset seed to ensure that the hyperparameters are independently random from data splits
    set.seed(NULL)
    seed.number = sample.int(10000, 1)[[1]]
    set.seed(seed.number)
    param <- list(objective = "reg:linear",
                  eval_metric = "rmse",
                  booster = sample(c('gblinear', 'gbtree'),1),
                  max_depth = sample(6:17, 1),
                  eta = runif(1, .01, .3),
                  gamma = runif(1, 0.0, 0.2), 
                  subsample = runif(1, .6, 1),
                  colsample_bytree = runif(1, .5, 1), 
                  min_child_weight = sample(1:40, 1),
                  max_delta_step = sample(1:10, 1)
    )
    cv.nround = 1000
    cv.nfold = 5
    
    ## ensure that the data splits in 5-fold cv is consistently the same for parameter tuning
    set.seed(2020)
    mdcv <- xgb.cv(data=dtrain, params = param, nthread=6, 
                   nfold=cv.nfold, nrounds=cv.nround,
                   verbose = F, early_stopping_rounds = 8, maximize=FALSE)
    
    min_rmse = min(mdcv$evaluation_log$test_rmse_mean)
    min_rmse_index = which.min(mdcv$evaluation_log$test_rmse_mean)
    
    if (min_rmse < best_rmse) {
      best_rmse = min_rmse
      best_rmse_index = min_rmse_index
      best_seednumber = seed.number
      best_iter = iter
      best_param = param
    }
  }
  
  nround = best_rmse_index
  md = xgb.train(data=dtrain, params=best_param, nrounds=nround, nthread=6)
  
  test_pred_studies = predict(md, newdata = dtest)

  test.df = test %>% 
    mutate(test_pred_studies = test_pred_studies,
           best_rmse = best_rmse, r_squared = cor(test_pred_studies, test_label)^2)
  return(test.df)
})
(proc.time() - start_time)
reg.results = bind_rows(results.list)
## not good results achieved



