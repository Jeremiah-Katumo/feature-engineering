# Install the necessary packages
library(tidyverse)
library(ggplot2)
library(visdat)
library(kableExtra)
library(caret)    
library(recipes)  
library(rsample)  

# create ames training data
set.seed(123)
data <- AmesHousing::make_ames()
split <- initial_split(data, prop = 0.7, strata = "Sale_Price")
data_train <- training(split)
data_test <- testing(split)

# TARGET ENGINEERING
transformed_response <- log(data_train$Sale_Price)

# Log transformation
data_recipe <- recipe(Sale_Price ~.,
                      data = data_train) %>%
  step_log(all_outcomes())
data_recipe

# Log transformation
train_log_y <- log(data_train$Sale_Price)
test_log_y <- log(data_train$Sale_Price)

# Box Cox transformation
lambda <- forecast::BoxCox.lambda(data_train$Sale_Price)
train_box_cox_y <- forecast::BoxCox(data_train$Sale_Price, lambda = lambda)
test_box_cox_y <- forecast::BoxCox(data_train$Sale_Price, lambda)

# Box Cox transform a value
y <- forecast::BoxCox(10, lambda)

# DEALING WITH MISSINGNESS
sum(is.na(AmesHousing::ames_raw))

# Using visdat package to visualize missing values
vis_miss(AmesHousing::ames_raw, cluster = TRUE)

# IMPUTATION
# Impute missing values onto ames_recipe for Gr_Liv_Area
data_recipe %>%
  step_medianimpute(Gr_Liv_Area)

# K-nearest neighbor imputation
data_recipe %>%
  step_knnimpute(all_predictors(), neighbors = 6)

# Tree-based imputation
data_recipe %>%
  step_impute_bag(all_predictors())  # or use step_bagimpute()

# FEATURE FILTERING
caret::nearZeroVar(data_train, saveMetrics = TRUE) %>%
  rownames_to_column() %>%
  filter(nzv)
# We can add step_zv() and step_nzv() to our ames_recipe to remove zero or
# near zero variance features.
data_recipe %>%
  step_nzv(all_predictors())

# NUMERIC FEATURE ENGINEERING
# 1) Skewness
# Normalize all numeric columns
recipe(Sale_Price ~., data = data_train) %>%
  step_YeoJohnson(all_numeric())

recipe(Sale_Price ~., data = data_train) %>%
  step_BoxCox(all_numeric())

# 2) Standardization
# Standardizing features includes centering and scaling so
# that numeric variables have zero mean and unit variance, which provides a
# common comparable unit of measure across all the variables.
data_recipe %>%
  step_center(all_numeric(), -all_outcomes()) %>%
  step_scale(all_numeric(), -all_outcomes())

# CATEGORICAL FEATURE ENGINEERING
# 1) Lumping
count(data_train, Neighborhood) %>% arrange(n)
count(data_train, Screen_Porch) %>% arrange(n)

# Lump levels for two features
lumping <- recipe(Sale_Price ~., data = data_train) %>%
  step_other(Neighborhood, threshold = 0.01,
             other = "other") %>%
  step_other(Screen_Porch, threshold = 0.1,
             other = ">0")

# 2) One-hot & dummy encoding
# By default, step_dummy() will create a full rank encoding but you can change this
# by setting one_hot = TRUE.
# Lump levels for two features
recipe(Sale_Price ~., data = data_train) %>%
  step_dummy(all_nominal(), one_hot = TRUE)

# If you have a data set with many categorical variables and those categorical variables 
# in turn have unique levels, the number of features can explode.
# In these cases you may want to explore label/ordinal encoding or some other alternatives
# 3) Label Encoding
# Original categories
count(data_train, MS_SubClass)

# Label encoded
recipe(Sale_Price ~., data = data_train) %>%
  step_integer(MS_SubClass) %>%
  prep(data_train) %>%
  bake(data_train) %>%
  count(MS_SubClass)

data_train %>%
  select(contains("Qual"))
# The various xxx_Qual features in the Ames housing are not ordered factors.
# For ordered factors you could also use step_ordinalscore().
# Original Categories
count(data_train, Overall_Qual)

# Label encoded
recipe(Sale_Price ~., data = data_train) %>%
  step_integer(Overall_Qual) %>%
  prep(data_train) %>%
  bake(data_train) %>%
  count(Overall_Qual)

# DIMENSION REDUCTION
recipe(Sale_Price ~., data = data_train) %>%
  step_center(all_numeric()) %>%
  step_scale(all_numeric()) %>%
  step_pca(all_numeric(), threshold = .95)

# PROPER IMPLEMENTATION
# Putting it together
blueprint <- recipe(Sale_Price ~., data = data_train) %>%
  step_nzv(all_nominal()) %>%
  step_integer(matches("Qual|Cond|QC|Qu")) %>%
  step_center(all_numeric(), -all_outcomes()) %>%
  step_scale(all_numeric(), -all_outcomes()) %>%
  step_pca(all_numeric(), -all_outcomes())
blueprint
# Next train the blueprint on some training data
prepare <- prep(blueprint, training = data_train)
prepare
# Lastly, we can apply our blueprint to new data (e.g., the training data or
# future test data) with bake().
bake_train <- bake(prepare, new_data = data_train)
bake_test <- bake(prepare, new_data = data_test)

# Consequently, the goal is to develop our blueprint, then within each resample
# iteration we want to apply prep() and bake() to our resample training and
# validation data.
blueprint <- recipe(Sale_Price ~., data = data_train) %>%
  step_nzv(all_nominal()) %>%
  step_integer(matches("Qual|Cond|QC|Qu")) %>%
  step_center(all_numeric(), -all_outcomes()) %>%
  step_scale(all_numeric(), -all_outcomes()) %>%
  step_dummy(all_nominal(), -all_outcomes(), one_hot = TRUE)

