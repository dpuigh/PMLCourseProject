# Practical Machine Learning
# Course Project

# Load libraries
library(plyr)
library(dplyr)
library(caret)

# enable multi-core processing
library(doParallel)

set.seed(1123)

# If files do not exist, download them
if( !(file.exists("pml-training.csv") && file.exists("pml-testing.csv")) ){
    print("Getting necessary files")
    fileURL <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
    download.file(fileURL, "pml-training.csv", method="curl")
    fileURL <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
    download.file(fileURL, "pml-testing.csv", method="curl")
    rm(fileURL)
}

train <- read.csv("pml-training.csv", na.strings=c("NA","#DIV/0!"))
test <- read.csv("pml-testing.csv", na.strings=c("NA","#DIV/0!"))

# Data Cleaning
#1 Check for NAs
listColNAs <- sapply(train,function(x) sum(is.na(x))/nrow(train)<0.90)

#2 Remove NAs and user related variables
train <- train[,listColNAs] %>%
    select(-(X:num_window))

# save for later, never touch until the end
test <- test[,listColNAs] %>%
    select(-(X:num_window))

# Split training sample again for training variables, 5000 events
forVarOpt <- sample(nrow(train), 5000)
trainVars <- train[forVarOpt,]
trainNoVars <- train[-forVarOpt,]

# Training and testing of different variables
inTrainVars <- createDataPartition( y=trainVars$classe, p=0.8, list=FALSE )
trainingVars <- trainVars[ inTrainVars, ]
testingVars <- trainVars[ -inTrainVars, ]

tc <- trainControl(method="cv", number=3)

# Use parallel processing to speed up training
cl <- makeCluster(detectCores())
registerDoParallel(cl)

print("Do first training! Be patient!")
modFit_Vars_rf <- train( classe ~ ., data=trainingVars, method="rf",
                         trainControl=tc )

stopCluster(cl)
# The stopCluster is necessary to terminate the extra processes

# Variable importance
imp <- varImp(modFit_Vars_rf)$importance
imp$name <- row.names(imp)
imp <- imp[order(imp$Overall,decreasing=TRUE),]

# Drop variables with importance below 2.0
dropVars <- imp[imp$Overall<2.0,]$name

redtrainingVars <- trainingVars[,!(names(trainingVars) %in% dropVars)]

# Train reduced variable model
cl <- makeCluster(detectCores())
registerDoParallel(cl)

print("Do second training with reduced number of variables! Be patient!")
modFit_redVars_rf <- train( classe ~ ., data=redtrainingVars, method="rf", 
                            trainControl=tc )
stopCluster(cl)

# Compare full and reduced variable models
cf_all <- confusionMatrix(testingVars$classe,predict(modFit_Vars_rf,testingVars))
cf_red <- confusionMatrix(testingVars$classe,predict(modFit_redVars_rf,testingVars))

results <- resamples(list(ALL=modFit_Vars_rf, RED=modFit_redVars_rf))
summary(results)
bwplot(results)

cf_all$table
cf_red$table

cf_all$overall
cf_red$overall

# Move forward with reduced set of variables
training_red <- trainNoVars[,!(names(trainNoVars) %in% dropVars)]

# 3-fold cross validation
folds <- createFolds(y=training_red$classe, k=3, list=TRUE)

train1 <- training_red[-folds[[1]],]
test1  <- training_red[ folds[[1]],]

train2 <- training_red[-folds[[2]],]
test2  <- training_red[ folds[[2]],]

train3 <- training_red[-folds[[3]],]
test3  <- training_red[ folds[[3]],]

# Train each of the folds, takes awhile
cl <- makeCluster(detectCores())
registerDoParallel(cl)

print("Do training on first fold! Be patient!")
modFit1_rf <- train( classe ~ ., data=train1, method="rf", trainControl=tc)

print("Do training on second fold! Be patient!")
modFit2_rf <- train( classe ~ ., data=train2, method="rf", trainControl=tc)

print("Do training on third fold! Be patient!")
modFit3_rf <- train( classe ~ ., data=train3, method="rf", trainControl=tc)

stopCluster(cl)

# Compare accuracies of the folds, take mean and sd
cf_1 <- confusionMatrix(test1$classe,predict(modFit1_rf,test1))
cf_2 <- confusionMatrix(test2$classe,predict(modFit2_rf,test2))
cf_3 <- confusionMatrix(test3$classe,predict(modFit3_rf,test3))

acc <- c(cf_1$overall[1], cf_2$overall[1], cf_2$overall[1])
acc
mean(acc)
sd(acc)


# Train final model, save 10% for final testing
finalTrain <- createDataPartition( y=training_red$classe, p=0.9, list=FALSE )
final_train <- training_red[ finalTrain, ]
final_test <- training_red[ -finalTrain, ]

cl <- makeCluster(detectCores())
registerDoParallel(cl)

print("Do final training! Be VERY patient!")
modFitFinal_rf <- train( classe ~ ., data=final_train, method="rf", trainControl=tc)

stopCluster(cl)

# Accuracy of model 
cf_final <- confusionMatrix(final_test$classe,predict(modFitFinal_rf,final_test))

cf_final$overall
cf_final$table

answers <- predict(modFitFinal_rf,test)
answers <- as.character(answers)

# Write out answers
source("pml_write_files.R")
pml_write_files(answers)
