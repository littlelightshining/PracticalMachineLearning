---
title: "Pratice Machine Learning"
author: "littlelight"
date: "03 of November of 2018"
output:
  html_document:
    keep_md: yes
---
## Synopsis

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset)

The training data (train) for this project are available here:

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv

The test data (test) are available here:

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv

The data for this project come from this source: http://groupware.les.inf.puc-rio.br/har.

The goal of this project is to predict the manner in which they did the exercise. This is the "classe" variable in the train set.

##### 1.SETTING WORKING DIRECTORY


```r
 setwd("C:/Users/carla/OneDrive/Coursera/Practical Machine Learning/Week4_Project")
```
##### 2.LOADING PACKAGES

```
## Loading required package: lattice
```

```
## Loading required package: ggplot2
```

```
## randomForest 4.6-14
```

```
## Type rfNews() to see new features/changes/bug fixes.
```

```
## 
## Attaching package: 'randomForest'
```

```
## The following object is masked from 'package:ggplot2':
## 
##     margin
```

```
## Loaded gbm 2.1.4
```
##### 3.READING DATA

```r
train<- read.csv("pml-training.csv", na.strings = c("NA", "#DIV/0!", ""))

test<- read.csv("pml-testing.csv", na.strings = c("NA", "#DIV/0!", ""))
```

##### 4.CREATING PARTITION TRAIN DATASET


```r
in.train <- createDataPartition(train$classe, p=0.7, list=FALSE)

train.set <- train[in.train, ]
test.set  <- train[-in.train, ]

dim(train.set)
```

```
## [1] 13737   160
```

```r
dim(test.set)
```

```
## [1] 5885  160
```
#### 5.REMOVE VARIABLES NEAR ZERO


```r
NZ <- nearZeroVar(train.set)
train.set <- train.set[, -NZ]
test.set <- test.set[, -NZ]

dim(train.set)
```

```
## [1] 13737   129
```

```r
dim(test.set)
```

```
## [1] 5885  129
```
#### 6.REMOVE VARIABLES MOSTLY NAs


```r
NAs <- sapply(train.set, function(x) mean(is.na(x)))>0.95
train.set <- train.set[, NAs==FALSE]
test.set <- test.set[, NAs==FALSE]

dim(train.set)
```

```
## [1] 13737    59
```

```r
dim(test.set)
```

```
## [1] 5885   59
```

#### 7. MODEL FIT TRAIN DATA SET - RANDOM FOREST


```r
set.seed(120)

control.rf <- trainControl(method="cv", number=3, verboseIter=FALSE)

modelfit.rf <- train(classe ~ ., data=train.set, method="rf",
                          trControl=control.rf)
modelfit.rf$finalModel
```

```
## 
## Call:
##  randomForest(x = x, y = y, mtry = param$mtry) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 41
## 
##         OOB estimate of  error rate: 0.03%
## Confusion matrix:
##      A    B    C    D    E  class.error
## A 3906    0    0    0    0 0.0000000000
## B    1 2656    1    0    0 0.0007524454
## C    0    1 2395    0    0 0.0004173623
## D    0    0    0 2252    0 0.0000000000
## E    0    0    0    1 2524 0.0003960396
```
#### 8. PREDICTION OF THE TEST DATA SET - RANDOM FOREST

```r
predict.rf <- predict(modelfit.rf, newdata=test.set)

conf.matrix.rf <- confusionMatrix(predict.rf, test.set$classe)

conf.matrix.rf
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1674    0    0    0    0
##          B    0 1139    0    0    0
##          C    0    0 1026    0    0
##          D    0    0    0  964    0
##          E    0    0    0    0 1082
## 
## Overall Statistics
##                                      
##                Accuracy : 1          
##                  95% CI : (0.9994, 1)
##     No Information Rate : 0.2845     
##     P-Value [Acc > NIR] : < 2.2e-16  
##                                      
##                   Kappa : 1          
##  Mcnemar's Test P-Value : NA         
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   1.0000   1.0000   1.0000   1.0000
## Specificity            1.0000   1.0000   1.0000   1.0000   1.0000
## Pos Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
## Neg Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Prevalence   0.2845   0.1935   0.1743   0.1638   0.1839
## Balanced Accuracy      1.0000   1.0000   1.0000   1.0000   1.0000
```
#### 9. PLOT RESULTS - RANDOM FOREST


```r
plot(conf.matrix.rf$table, col = conf.matrix.rf$byClass, 
     main = paste("Random Forest - Accuracy =",
                  round(conf.matrix.rf$overall['Accuracy'], 4)))
```

![](PML_files/figure-html/plotresultsrandomforest-1.png)<!-- -->

#### 10. MODEL FIT TRAIN DATA SET - GBM


```r
set.seed(120)

control.gmb <- trainControl(method = "repeatedcv", number = 5, repeats = 1)

model.fit.gmb <- train(classe ~ ., data=train.set, method = "gbm",
                    trControl = control.gmb, verbose = FALSE)

model.fit.gmb$finalModel
```

```
## A gradient boosted model with multinomial loss function.
## 50 iterations were performed.
## There were 80 predictors of which 1 had non-zero influence.
```
#### 11. MODEL FIT TEST DATA SET - GBM


```r
predict.gbm <- predict(model.fit.gmb, newdata=test.set)

conf.matrix.gbm <- confusionMatrix(predict.gbm, test.set$classe)

conf.matrix.gbm
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1674    0    0    0    0
##          B    0 1139    0    0    0
##          C    0    0 1026    0    0
##          D    0    0    0  964    0
##          E    0    0    0    0 1082
## 
## Overall Statistics
##                                      
##                Accuracy : 1          
##                  95% CI : (0.9994, 1)
##     No Information Rate : 0.2845     
##     P-Value [Acc > NIR] : < 2.2e-16  
##                                      
##                   Kappa : 1          
##  Mcnemar's Test P-Value : NA         
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   1.0000   1.0000   1.0000   1.0000
## Specificity            1.0000   1.0000   1.0000   1.0000   1.0000
## Pos Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
## Neg Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Prevalence   0.2845   0.1935   0.1743   0.1638   0.1839
## Balanced Accuracy      1.0000   1.0000   1.0000   1.0000   1.0000
```
#### 12. PLOT RESULTS - GBM


```r
plot(conf.matrix.gbm$table, col = conf.matrix.gbm$byClass, 
     main = paste("GBM - Accuracy =", round(conf.matrix.gbm$overall['Accuracy'], 4)))
```

![](PML_files/figure-html/plotresultsgbm-1.png)<!-- -->

#### 13. SELECT AND APPLY THE MODEL TO TEST DATA


#####Based on the accuracy results we selected Random Forest model to be applied to test data.


```r
predict.test <- predict(modelfit.rf, newdata=test)

predict.test
```

```
##  [1] A A A A A A A A A A A A A A A A A A A A
## Levels: A B C D E
```



