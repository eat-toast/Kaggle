setwd('C:\\Data_analysis\\kaggle\\titanic')
train<- read.csv('train.csv', stringsAsFactors = FALSE)
test<-  read.csv('test.csv', stringsAsFactors = FALSE)
submission<- read.csv("sample_submission.csv")
library(stringr)
library(caret)
library(randomforest)
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


# Age
sum(is.na( titanic.full$Age))
age.median<- median(titanic.full$Age, na.rm= TRUE) 
titanic.full[ is.na( titanic.full$Age), 'Age'  ] <- age.median


# Fare
sum(is.na( titanic.full$Fare))
fare.median<- median(titanic.full$Fare, na.rm= TRUE) 
titanic.full[ is.na( titanic.full$Fare), 'Fare'  ] <- fare.median

# Name 왕십리에서 추가하기.


# categorical casting
titanic.full$Pclass <- as.factor(titanic.full$Pclass)
titanic.full$Sex <- as.factor(titanic.full$Sex)
titanic.full$Embarked <- as.factor(titanic.full$Embarked)


# Split
titanic.train<- titanic.full[titanic.full$is_train==TRUE, ]
titanic.test<- titanic.full[titanic.full$is_train==FALSE, ]

# 생존 변수
titanic.train$Survived<- as.factor(titanic.train$Survived)

# random Forests
''
fit_random_forest<- randomForest(formula = Survived ~ Pclass + Sex + Age +  SibSp + Parch + Fare + Embarked
                                 , data = titanic.train)

pred<- predict(fit_random_forest, titanic.test)

submission$Survived<- pred

write.csv(submission, '20190101.csv', row.names = FALSE)
