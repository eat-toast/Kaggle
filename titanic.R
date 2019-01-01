library(dplyr)
library(stringr)
library(caret)
library(randomForest)

setwd('D:\\타이타닉')

train<- read.csv('train.csv', stringsAsFactors = FALSE)
test<-  read.csv('test.csv', stringsAsFactors = FALSE)
submission<- read.csv("sample_submission.csv")


# 전처리를위해 데이터 하나로 묶기
train$is_train<- TRUE
test$is_train<- FALSE


# test에 생존 변수 추가 
test$Survived<- NA

# full 데이터 생성
titanic.full <- rbind(train, test)

# Embarkes
table(titanic.full$Embarked)
titanic.full[ titanic.full$Embarked == '','Embarked'  ] <- 'S'


# Fare
sum(is.na( titanic.full$Fare))
fare.median<- median(titanic.full$Fare, na.rm= TRUE) 
titanic.full[ is.na( titanic.full$Fare), 'Fare'  ] <- fare.median



# title(Name)
titles<- sapply(titanic.full$Name, function(x) unlist(str_extract_all(x, '([A-za-z]+)\\w\\.')))
names(titles)<-NULL
sort(table(titles))

title_dic<- c("Capt.", "Col.", "Major.", "Rev.", "Dr.", "Miss.", "Mlle.", "Mrs.", "Mme."
              , "Ms.", "Mr.", "Master.", "Lady.", "Jonkheer.", "Countess.", "Don.", "Sir.", "Dona.")

names(title_dic)<-c("Officer", "Officer", "Officer", "Officer", "Officer", "Miss", "Miss", "Mrs", "Mrs"
                    , "Mrs", "Mr", "Master", "Royalty", "Royalty", "Royalty", "Royalty", "Royalty", "Royalty")

# make title
titanic.full$title<-0
for(i in 1:nrow(titanic.full)){
  idx <- which(titles[i] == title_dic)
  titanic.full[i, 'title']<- names(title_dic)[idx]
}


# Age
# Age에 Na가 없는 데이터를 기반으로 평균 나이 측정
mean_Ages = titanic.full%>%filter(! is.na(Age))%>%group_by(title)%>%summarize(mean_age = mean(Age))
Master_age<- mean_Ages[mean_Ages$title == 'Master', "mean_age"]
Miss_age<- mean_Ages[mean_Ages$title == 'Miss', "mean_age"]
Mr_age<- mean_Ages[mean_Ages$title == 'Mr', "mean_age"]
Mrs_age<- mean_Ages[mean_Ages$title == 'Mrs', "mean_age"]
Officer_age<- mean_Ages[mean_Ages$title == 'Officer', "mean_age"]
Royalty_age<- mean_Ages[mean_Ages$title == 'Royalty', "mean_age"]

titanic.full[titanic.full$title=='Master' & is.na(titanic.full$Age), 'Age']<- Master_age
titanic.full[titanic.full$title=='Miss' & is.na(titanic.full$Age), 'Age']<- Miss_age
titanic.full[titanic.full$title=='Mr' & is.na(titanic.full$Age), 'Age']<- Mr_age
titanic.full[titanic.full$title=='Mrs' & is.na(titanic.full$Age), 'Age']<- Mrs_age
titanic.full[titanic.full$title=='Officer' & is.na(titanic.full$Age), 'Age']<- Officer_age
titanic.full[titanic.full$title=='Royalty' & is.na(titanic.full$Age), 'Age']<- Royalty_age

# categorical casting
titanic.full$Pclass <- as.factor(titanic.full$Pclass)
titanic.full$Sex <- as.factor(titanic.full$Sex)
titanic.full$Embarked <- as.factor(titanic.full$Embarked)
titanic.full$title <- as.factor(titanic.full$title)


# Split
titanic.train<- titanic.full[titanic.full$is_train==TRUE, ]
titanic.test<- titanic.full[titanic.full$is_train==FALSE, ]



# 생존 변수
titanic.train$Survived<- as.factor(titanic.train$Survived)



# random Forests
fit_random_forest<- randomForest(formula = Survived ~ title + Pclass + Sex + Age 
                                 + SibSp + Parch + Fare + Embarked
                                 , data = titanic.train)


pred<- predict(fit_random_forest, titanic.test)
submission$Survived<- pred

write.csv(submission, '20190101_2.csv', row.names = FALSE)

# h2o
library(h2oEnsemble)  # This will load the `h2o` R package as well
h2o.init(nthreads = -1)  # Start an H2O cluster with nthreads = num cores on your machine
h2o.removeAll() # (Optional) Remove all objects in H2O cluster

idx<- createDataPartition(titanic.train$Survived, p=.8, list=FALSE)

# 불필요 변수 제거
feature_name<- setdiff(names(titanic.train), c('PassengerId','Ticket', 'Cabin', 'is_train'))


train<- titanic.train#[idx, ]
test<- titanic.test#[-idx, ]

train<- as.h2o(train)
test<- as.h2o(test)


# Identify predictors and response
y <- "Survived"
x <- setdiff(feature_name, y)



learner <- c("h2o.glm.wrapper", "h2o.randomForest.wrapper", 
             "h2o.gbm.wrapper", "h2o.deeplearning.wrapper")
metalearner <- "h2o.gbm.wrapper"

family = 'binomial'
fit <- h2o.ensemble(x = x, y = y, 
                    training_frame = train, 
                    family = family, 
                    learner = learner, 
                    metalearner = metalearner,
                    cvControl = list(V = 5))



(perf <- h2o.ensemble_performance(fit, newdata = test))
print(perf, metric = "AUTO")


# feature_name<- setdiff(feature_name, 'Survived')
pred <- predict(fit, newdata = test[,feature_name])
predictions <- as.data.frame(pred$pred)[,1]  #third column is P(Y==1)
submission[,2]<- predictions

write.csv(submission, '20190101_3.csv', row.names = FALSE)






h2o.glm.1 <- function(..., alpha = 0.0) h2o.glm.wrapper(..., alpha = alpha)
h2o.glm.2 <- function(..., alpha = 0.5) h2o.glm.wrapper(..., alpha = alpha)
h2o.glm.3 <- function(..., alpha = 1.0) h2o.glm.wrapper(..., alpha = alpha)
h2o.randomForest.1 <- function(..., ntrees = 200, nbins = 50, seed = 1) h2o.randomForest.wrapper(..., ntrees = ntrees, nbins = nbins, seed = seed)
h2o.randomForest.2 <- function(..., ntrees = 200, sample_rate = 0.75, seed = 1) h2o.randomForest.wrapper(..., ntrees = ntrees, sample_rate = sample_rate, seed = seed)
h2o.randomForest.3 <- function(..., ntrees = 200, sample_rate = 0.85, seed = 1) h2o.randomForest.wrapper(..., ntrees = ntrees, sample_rate = sample_rate, seed = seed)
h2o.randomForest.4 <- function(..., ntrees = 200, nbins = 50, balance_classes = TRUE, seed = 1) h2o.randomForest.wrapper(..., ntrees = ntrees, nbins = nbins, balance_classes = balance_classes, seed = seed)
h2o.gbm.1 <- function(..., ntrees = 100, seed = 1) h2o.gbm.wrapper(..., ntrees = ntrees, seed = seed)
h2o.gbm.2 <- function(..., ntrees = 100, nbins = 50, seed = 1) h2o.gbm.wrapper(..., ntrees = ntrees, nbins = nbins, seed = seed)
h2o.gbm.3 <- function(..., ntrees = 100, max_depth = 10, seed = 1) h2o.gbm.wrapper(..., ntrees = ntrees, max_depth = max_depth, seed = seed)
h2o.gbm.4 <- function(..., ntrees = 100, col_sample_rate = 0.8, seed = 1) h2o.gbm.wrapper(..., ntrees = ntrees, col_sample_rate = col_sample_rate, seed = seed)
h2o.gbm.5 <- function(..., ntrees = 100, col_sample_rate = 0.7, seed = 1) h2o.gbm.wrapper(..., ntrees = ntrees, col_sample_rate = col_sample_rate, seed = seed)
h2o.gbm.6 <- function(..., ntrees = 100, col_sample_rate = 0.6, seed = 1) h2o.gbm.wrapper(..., ntrees = ntrees, col_sample_rate = col_sample_rate, seed = seed)
h2o.gbm.7 <- function(..., ntrees = 100, balance_classes = TRUE, seed = 1) h2o.gbm.wrapper(..., ntrees = ntrees, balance_classes = balance_classes, seed = seed)
h2o.gbm.8 <- function(..., ntrees = 100, max_depth = 3, seed = 1) h2o.gbm.wrapper(..., ntrees = ntrees, max_depth = max_depth, seed = seed)
h2o.deeplearning.1 <- function(..., hidden = c(500,500), activation = "Rectifier", epochs = 50, seed = 1)  h2o.deeplearning.wrapper(..., hidden = hidden, activation = activation, seed = seed)
h2o.deeplearning.2 <- function(..., hidden = c(200,200,200), activation = "Tanh", epochs = 50, seed = 1)  h2o.deeplearning.wrapper(..., hidden = hidden, activation = activation, seed = seed)
h2o.deeplearning.3 <- function(..., hidden = c(500,500), activation = "RectifierWithDropout", epochs = 50, seed = 1)  h2o.deeplearning.wrapper(..., hidden = hidden, activation = activation, seed = seed)
h2o.deeplearning.4 <- function(..., hidden = c(500,500), activation = "Rectifier", epochs = 50, balance_classes = TRUE, seed = 1)  h2o.deeplearning.wrapper(..., hidden = hidden, activation = activation, balance_classes = balance_classes, seed = seed)
h2o.deeplearning.5 <- function(..., hidden = c(100,100,100), activation = "Rectifier", epochs = 50, seed = 1)  h2o.deeplearning.wrapper(..., hidden = hidden, activation = activation, seed = seed)
h2o.deeplearning.6 <- function(..., hidden = c(50,50), activation = "Rectifier", epochs = 50, seed = 1)  h2o.deeplearning.wrapper(..., hidden = hidden, activation = activation, seed = seed)
h2o.deeplearning.7 <- function(..., hidden = c(100,100), activation = "Rectifier", epochs = 50, seed = 1)  h2o.deeplearning.wrapper(..., hidden = hidden, activation = activation, seed = seed)

learner <- c("h2o.glm.1","h2o.glm.2","h2o.glm.3",
             "h2o.randomForest.1", "h2o.randomForest.2",'h2o.randomForest.3','h2o.randomForest.4',
             "h2o.gbm.1","h2o.gbm.2","h2o.gbm.3","h2o.gbm.4","h2o.gbm.5","h2o.gbm.6",'h2o.gbm.7',"h2o.gbm.8",
             "h2o.deeplearning.1","h2o.deeplearning.2","h2o.deeplearning.3",'h2o.deeplearning.4',
             "h2o.deeplearning.5","h2o.deeplearning.6", "h2o.deeplearning.7")

