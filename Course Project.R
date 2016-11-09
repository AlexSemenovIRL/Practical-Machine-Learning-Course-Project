rm(list = ls())
setwd("C:/Users/asemenov/Desktop/Machine Learning Course project")

library(caret)
library(randomForest)

set.seed(666)
#TrainSet <- read.csv(url("http://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"), na.strings=c("NA",""), header=T)
#TestSet <- read.csv(url("http://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"), na.strings=c("NA",""), header=T)

TrainSet <- read.csv("pml-training.csv", na.strings=c("NA",""), header=T)
TestSet <- read.csv("pml-testing.csv", na.strings=c("NA",""), header=T)

# Compare the Traiing and Test Sets for Column Names, excluding the last column
all(names(TrainSet[1:length(TrainSet)-1]) == names(TestSet[1:length(TestSet)-1]))


# Identify near zero variance predictors
nzv <- nearZeroVar(TrainSet, saveMetrics=TRUE)
TrainSet <- TrainSet[,nzv$nzv==FALSE]


# Split the Initial Training set into Training and validation set 70/30
Partition <- createDataPartition(TrainSet$classe, p = 0.7, list=FALSE)
TrainingSet <- TrainSet[Partition,]
ValidationSet <- TrainSet[-Partition,]


# Remove variables that are most NAs, here we use 95% of them should be NA
RemoveNA <- sapply(TrainingSet, function(x) mean(is.na(x))) > 0.95
TrainingSet <- TrainingSet[,RemoveNA==F]
ValidationSet <- ValidationSet[,RemoveNA==F]


# Take out the variables that are not to be used
TrainingSet <- TrainingSet[, -(1:7)]
ValidationSet <- ValidationSet[, -(1:7)]

# Model #1 Using Random Forests
set.seed(666)
RF_Model <- randomForest(classe ~ ., data = TrainingSet)
RF_Prediction <- predict(RF_Model, ValidationSet, type = "class")
RF_ConfusionMatrix <- confusionMatrix(RF_Prediction, ValidationSet$classe)
RF_ConfusionMatrix

# Plot the error rate as a function of # of trees for RF Model
plot(RF_Model);box();grid()

RF_Importance <- data.frame(Variable = names(RF_Model$importance[,1]), Importance = RF_Model$importance[,1])
RF_Importance <- RF_Importance[order(-RF_Importance[,2]),]
RF_Importance

# Model #2 GBM Model
set.seed(666)
Control <- trainControl(method = "cv",2)
GBM_Model <- train(classe ~ ., data = TrainingSet, method = "gbm",trControl = Control,verbose = FALSE)

GBM_Prediction <- predict(GBM_Model, newdata = ValidationSet)
GBM_ConfusionMatrix <- confusionMatrix(GBM_Prediction, ValidationSet$classe)
GBM_ConfusionMatrix

# Plot the error rate as a function of # of trees for GBM Model
plot(GBM_Model)


# Compare the two models
RF_Summary <- round(RF_ConfusionMatrix$overall,4)
GBM_Summary <- round(GBM_ConfusionMatrix$overall,4)

# Model Comparison
ModelComparison <- rbind(RF_Summary,GBM_Summary)
ModelComparison[,1:4]


# Evaluating two models we can see that the Random Forest Model is better
Result <- predict(RF_Model, TestSet[, -length(names(TestSet))])
Result


