# MLAssignment
Assignment

library(caret); library(randomForest); library(rpart)
## Loading required package: lattice
## Loading required package: ggplot2
## randomForest 4.6-12
## Type rfNews() to see new features/changes/bug fixes.
url.train <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
url.test <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
training <- read.csv(url(url.train), na.strings = c("NA", "", "#DIV0!"))
testing <- read.csv(url(url.test), na.strings = c("NA", "", "#DIV0!"))
We need to verify that the columns in both tables (training, testing) are the same or not.

#define the same columns
sameColumsName <- colnames(training) == colnames(testing)
colnames(training)[sameColumsName==FALSE]
## [1] "classe"
It is obvious that the information about the “classe” is not included in the testing data.

Cleaning training & testing data
We can see several columns not relevant for predicting and to the activity movement. Delete columns with all missing values.

training<-training[,colSums(is.na(training)) == 0]
testing <-testing[,colSums(is.na(testing)) == 0]
#dimTraining <- dim(training)
#dimTesting <- dim(testing)
head(colnames(training), 10)
##  [1] "X"                    "user_name"            "raw_timestamp_part_1"
##  [4] "raw_timestamp_part_2" "cvtd_timestamp"       "new_window"          
##  [7] "num_window"           "roll_belt"            "pitch_belt"          
## [10] "yaw_belt"
training <- training[,8:dim(training)[2]]
testing <- testing[,8:dim(testing)[2]]
We can delete first 7 variables, because they are irrelevant to our project: “user_name”, “raw_timestamp_part_1”, “raw_timestamp_part_2”, “cvtd_timestamp”, “new_window” and “num_window” (columns 1 to 7).

Activity model
CLASSE is our outcome variable (5-level factor variable).
“Participants were asked to perform one set of 10 repetitions of the Unilateral Dumbbell Biceps Curl in five different fashions:
(Class A) - exactly according to the specification
(Class B) - throwing the elbows to the front
(Class C) - lifting the dumbbell only halfway
(Class D) - lowering the dumbbell only halfway
(Class E) - throwing the hips to the front
Class A corresponds to the specified execution of the exercise, while the other 4 classes correspond to common mistakes. The exercises were performed by six male participants aged between 20-28 years, with little weight lifting experience. We made sure that all participants could easily simulate the mistakes in a safe and controlled manner by using a relatively light dumbbell (1.25kg)” [1].
Prediction evaluations will be based on maximizing the accuracy and minimizing the out-of-sample error. To predict we will use all the variables after cleaning.
“Tree” & “Random Forest” will be apply as different learning methods. The model with the highest accuracy will be choosen as our final model.

Training and CrossValidation (data slicing)
Our outcome variable classe is an unordered factor variable. Thus, we can choose our error type as 1-accuracy. We have a large sample size (19622) in the Training data set. This allow us to divide our Training sample into TrainingCV and testingCV to allow cross-validation. Decision tree and random forest algorithms are known for their ability of detecting the features that are very important for classification.

Cross-validation will be performed by subsampling our training data and splitted into training part and cross-validation with ratio 0.7. The most accurate model will be choosen and tested on the original Testing dataset.

set.seed(12345)
inTrain <- createDataPartition(training$classe, p=0.7, list=FALSE)
trainingCV <- training[inTrain,]
testingCV <- training[-inTrain,]
dim(trainingCV); dim(testingCV)
## [1] 13737    53
## [1] 5885   53
Plot some data
Plotting some accelaration data in trainingCV data, we can see that the pattern is very similar and hard to distinguish among those classes A,B,C,D,E

qplot(accel_arm_x, accel_arm_y, col=classe, data=trainingCV)


qplot(accel_forearm_x, accel_forearm_y, col=classe, data=trainingCV)


#qplot(accel_dumbbell_x, accel_dumbbell_y, col=classe, data=trainingSlice)
Predicting models
Apply Classification Tree model
modelCTree <- rpart(classe ~ ., data=trainingCV, method="class")
predictionCTree <- predict(modelCTree, testingCV, type="class")
CTree <- confusionMatrix(predictionCTree, testingCV$classe)
CTree
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1498  196   69  106   25
##          B   42  669   85   86   92
##          C   43  136  739  129  131
##          D   33   85   98  553   44
##          E   58   53   35   90  790
## 
## Overall Statistics
##                                           
##                Accuracy : 0.722           
##                  95% CI : (0.7104, 0.7334)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.6467          
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.8949   0.5874   0.7203  0.57365   0.7301
## Specificity            0.9060   0.9357   0.9097  0.94717   0.9509
## Pos Pred Value         0.7909   0.6869   0.6273  0.68020   0.7700
## Neg Pred Value         0.9559   0.9043   0.9390  0.91897   0.9399
## Prevalence             0.2845   0.1935   0.1743  0.16381   0.1839
## Detection Rate         0.2545   0.1137   0.1256  0.09397   0.1342
## Detection Prevalence   0.3218   0.1655   0.2002  0.13815   0.1743
## Balanced Accuracy      0.9004   0.7615   0.8150  0.76041   0.8405
library(rpart.plot)
rpart.plot(modelCTree)


Apply Random forest model
modelRF <- randomForest(classe ~ ., data=trainingCV, method="class")
predictionRF <- predict(modelRF, testingCV, type="class")
RF <- confusionMatrix(predictionRF, testingCV$classe)
RF
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1673    9    0    0    0
##          B    1 1127   13    0    0
##          C    0    3 1011   14    0
##          D    0    0    2  949    5
##          E    0    0    0    1 1077
## 
## Overall Statistics
##                                          
##                Accuracy : 0.9918         
##                  95% CI : (0.9892, 0.994)
##     No Information Rate : 0.2845         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.9897         
##  Mcnemar's Test P-Value : NA             
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9994   0.9895   0.9854   0.9844   0.9954
## Specificity            0.9979   0.9971   0.9965   0.9986   0.9998
## Pos Pred Value         0.9946   0.9877   0.9835   0.9927   0.9991
## Neg Pred Value         0.9998   0.9975   0.9969   0.9970   0.9990
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2843   0.1915   0.1718   0.1613   0.1830
## Detection Prevalence   0.2858   0.1939   0.1747   0.1624   0.1832
## Balanced Accuracy      0.9986   0.9933   0.9909   0.9915   0.9976
CV <- testingCV
CV$GOODpred <- testingCV$classe == predictionRF
qplot(accel_forearm_x, accel_forearm_y, col=GOODpred, data=CV)


On the plot you can see the new prediction error values. Due to the high degree of accuracy, you can see that the point of failure are poor.

Accuracy & Expected out-of-sample error
Accuracy is the proportion of correct classified observation over the total sample in the CrossValidation data set. Look a comparison of the both methods. Random Forest method is much much better.

Accuracy	Out-of-Sample Error
Classification Tree Method	0.7220051	0.2779949
Random Forest Method	0.9918437	0.0081563
Final Prediction
FinalPrediction <- predict(modelRF, testing)
kable(t(data.frame(FinalPrediction)))
1	2	3	4	5	6	7	8	9	10	11	12	13	14	15	16	17	18	19	20
FinalPrediction	B	A	B	A	A	E	D	B	A	A	B	C	B	A	E	E	A	B	B	B
Conclusions: Prediction evaluations were based on maximizing the accuracy and minimizing the out-of-sample error. All other available variables after cleaning were used for prediction. Two models were tested using decision tree and random forest algorithms. The model with the highest accuracy were chosen as final model.

# Write files for the final prediction
pml_files = function(x){ 
  for(i in 1:length(x)) 
        { 
        filename = paste0("problem_",i,".txt") 
        write.table(x[i],file=filename, row.names=FALSE, col.names=FALSE, quote=FALSE)
        }
}

pml_files(FinalPrediction)
