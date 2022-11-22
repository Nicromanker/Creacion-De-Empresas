library(tidyverse)
library(tidymodels)
library(parallel)
library(doParallel)
library(rpart.plot)
library(rpart)

#Si estas leyendo este comentario, gracias :)

setwd('C:/Users/nicoc/GIT/Creacion-De-Empresas')

datos = read.csv("UCI_Credit_Card.csv")

str(datos)

datos = datos %>% rename(default = default.payment.next.month) %>% 
                  mutate(SEX = as.factor(SEX),
                         EDUCATION = as.factor(EDUCATION),
                         MARRIAGE = as.factor(MARRIAGE),
                         default = as.factor(default)) %>% 
                  select(-c(ID))

set.seed(42069)

data_split     =   initial_split(datos, prop = 0.80, strata = default)
data_train     =   data_split %>% training()
data_test      =   data_split %>% testing()

table(data_train$default) / length(data_train$default)
table(data_test$default) / length(data_test$default)

data_folds =  vfold_cv(data_train , v = 10)

receta = recipe(default ~ .,
                data = data_train)

modelo = decision_tree(cost_complexity = tune(),
                       tree_depth = tune(),
                       min_n = tune()) %>% 
         set_engine('rpart') %>% 
         set_mode('classification')

grid_tree = expand_grid(cost_complexity = seq(-10,-1,by=1),
                        tree_depth=c(1,2,3,4),
                        min_n = c(20,30,40))


cores = detectCores()

workflow_tree_tune = workflow() %>% 
                     add_model(modelo) %>% 
                     add_recipe(receta)

cl <- parallel::makeCluster(cores)

doParallel::registerDoParallel(cl)

cv_tree = workflow_tree_tune %>%
            tune_grid(data_folds,
            grid = grid_tree,
            metrics = metric_set(roc_auc))

parallel::stopCluster(cl)

best_tree = cv_tree %>% select_best("roc_auc")

best_tree

final_tree = workflow_tree_tune %>% finalize_workflow(best_tree)

tree_fit = final_tree %>% fit(data = data_train)

prediccion_test = predict(tree_fit, new_data = data_test)

data_test$pred = as.factor(prediccion_test$.pred_class)

matriz_confusion = conf_mat(data_test, truth = default, estimate = pred)

arbol_grafico = tree_fit %>% extract_fit_engine()

rpart.plot(arbol_grafico, type = 5)