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
## corrplot 0.84 loaded
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
## [1] 13737   131
```

```r
dim(test.set)
```

```
## [1] 5885  131
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
#### 7. REMOVE IDENTIFICATION ONLY VARIABLES

```r
train.set <- train.set[, -(1:5)]
test.set  <- test.set[, -(1:5)]
dim(train.set)
```

```
## [1] 13737    54
```

```r
dim(test.set)
```

```
## [1] 5885   54
```

#### 8. CORRELATION ANALYSIS


```r
corr.matrix <- cor(train.set[, -54])

corrplot(corr.matrix, order = "FPC", method = "color", type = "lower", 
         tl.cex = 0.8, tl.col = rgb(0, 0, 0))
```

![](PML_files/figure-html/correlationanalysis-1.png)<!-- -->

#### 9. MODEL FIT TRAIN DATA SET - RANDOM FOREST


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
## No. of variables tried at each split: 27
## 
##         OOB estimate of  error rate: 0.25%
## Confusion matrix:
##      A    B    C    D    E  class.error
## A 3904    1    0    0    1 0.0005120328
## B    4 2649    4    1    0 0.0033860045
## C    0    4 2392    0    0 0.0016694491
## D    0    0   13 2238    1 0.0062166963
## E    0    0    0    5 2520 0.0019801980
```
#### 10. PREDICTION OF THE TEST DATA SET - RANDOM FOREST

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
##          A 1674    5    0    0    0
##          B    0 1133    2    0    0
##          C    0    1 1024    1    0
##          D    0    0    0  963    1
##          E    0    0    0    0 1081
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9983          
##                  95% CI : (0.9969, 0.9992)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9979          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   0.9947   0.9981   0.9990   0.9991
## Specificity            0.9988   0.9996   0.9996   0.9998   1.0000
## Pos Pred Value         0.9970   0.9982   0.9981   0.9990   1.0000
## Neg Pred Value         1.0000   0.9987   0.9996   0.9998   0.9998
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2845   0.1925   0.1740   0.1636   0.1837
## Detection Prevalence   0.2853   0.1929   0.1743   0.1638   0.1837
## Balanced Accuracy      0.9994   0.9972   0.9988   0.9994   0.9995
```
#### 11. PLOT RESULTS - RANDOM FOREST


```r
plot(conf.matrix.rf$table, col = conf.matrix.rf$byClass, 
     main = paste("Random Forest - Accuracy =",
                  round(conf.matrix.rf$overall['Accuracy'], 4)))
```

![](PML_files/figure-html/plotresultsrandomforest-1.png)<!-- -->

#### 12. MODEL FIT TRAIN DATA SET - GBM


```r
set.seed(120)

control.gmb <- trainControl(method = "repeatedcv", number = 5, repeats = 1)

model.fit.gmb <- train(classe ~ ., data=train.set, method = "gbm",
                    trControl = control.gmb, verbose = FALSE)

model.fit.gmb$finalModel
```

```
## A gradient boosted model with multinomial loss function.
## 150 iterations were performed.
## There were 53 predictors of which 41 had non-zero influence.
```
#### 13. MODEL FIT TEST DATA SET - GBM


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
##          A 1668    9    0    0    0
##          B    6 1123    6    6    4
##          C    0    7 1013    7    1
##          D    0    0    6  951    6
##          E    0    0    1    0 1071
## 
## Overall Statistics
##                                           
##                Accuracy : 0.99            
##                  95% CI : (0.9871, 0.9924)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9873          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9964   0.9860   0.9873   0.9865   0.9898
## Specificity            0.9979   0.9954   0.9969   0.9976   0.9998
## Pos Pred Value         0.9946   0.9808   0.9854   0.9875   0.9991
## Neg Pred Value         0.9986   0.9966   0.9973   0.9974   0.9977
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2834   0.1908   0.1721   0.1616   0.1820
## Detection Prevalence   0.2850   0.1946   0.1747   0.1636   0.1822
## Balanced Accuracy      0.9971   0.9907   0.9921   0.9920   0.9948
```
#### 14. PLOT RESULTS - GBM


```r
plot(conf.matrix.gbm$table, col = conf.matrix.gbm$byClass, 
     main = paste("GBM - Accuracy =", round(conf.matrix.gbm$overall['Accuracy'], 4)))
```

![](PML_files/figure-html/plotresultsgbm-1.png)<!-- -->

#### 15. SELECT AND APPLY THE MODEL TO TEST DATA


#####Based on the accuracy results we selected Random Forest model to be applied to test data.


```r
predict.test <- predict(modelfit.rf, newdata=test)

predict.test
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```



