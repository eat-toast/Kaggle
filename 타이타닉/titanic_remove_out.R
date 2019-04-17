library(dplyr)
library(stringr)
library(caret)
library(randomForest)


setwd('D:\\타이타닉\\Kaggle')

train<- read.csv('train.csv', stringsAsFactors = FALSE)
test<-  read.csv('test.csv', stringsAsFactors = FALSE)
submission<- read.csv("sample_submission.csv")

# 아웃라이어 제거 
out_idx<- c(28, 89, 160, 181, 202, 325, 342, 793, 847, 864)
train<- train[-out_idx, ]

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
titanic.full$Fare<-log(titanic.full$Fare+1)


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

# Family size
titanic.full$family_size <-titanic.full$SibSp + titanic.full$Parch+1

# 이름의 길이 (X)
titanic.full$name_len <- nchar(titanic.full$Name)

# isAlone
titanic.full$isAlone <- ifelse(titanic.full$family_size==1, 1, 0)


# Cabin 선실에 대한 정보 출처: https://www.kaggle.com/c/titanic/discussion/4693
Cabin<- titanic.full$Cabin
cabin<- vector('character', length(Cabin))
for(i in 1:length(Cabin)){
  if(nchar(Cabin[i]) == 0 ){cabin[i]<-'U'}else{
    cabin[i]<- substr(Cabin[i],1,1)
  }
}
# T는 A로 분류
T_idx<- which(cabin == 'T')
cabin[T_idx]<- 'A'

cabin_dic<- c("A", 'B', 'C', 'D', 'E', 'F', 'G', 'U')


# make new_cabin
titanic.full$new_Cabin<- cabin


# categorical casting
titanic.full$Pclass <- as.factor(titanic.full$Pclass)
titanic.full$Sex <- as.factor(titanic.full$Sex)
titanic.full$Embarked <- as.factor(titanic.full$Embarked)
titanic.full$title <- as.factor(titanic.full$title)
titanic.full$isAlone<- as.factor(titanic.full$isAlone)
titanic.full$new_Cabin<- as.factor(titanic.full$new_Cabin)

# Split
titanic.train<- titanic.full[titanic.full$is_train==TRUE, ]
titanic.test<- titanic.full[titanic.full$is_train==FALSE, ]




# 생존 변수
titanic.train$Survived<- as.factor(titanic.train$Survived)




# random Forests
fit_random_forest<- randomForest(formula = Survived ~ title + Pclass + Sex + Age 
                                 + family_size + Fare+isAlone+new_Cabin  #(SibSp + Parch +Embarked )
                                 
                                 , data = titanic.train)


pred<- predict(fit_random_forest, titanic.test)
submission$Survived<- pred

write.csv(submission, '20190113_8.csv', row.names = FALSE)

# thresh hold 구하기
prob<- predict(fit_random_forest, titanic.train, type= 'prob')[,1]

Thresh_hold<- sort(unique(prob))

t<-1
recall<- vector('numeric', length(Thresh_hold))
precision<- vector('numeric', length(Thresh_hold))
actual=titanic.train$Survived

while(TRUE){
  thresh_hold<- mean(Thresh_hold[t] + Thresh_hold[t+1])
  temp_pred <- prob < thresh_hold
  
  pred = as.factor(as.numeric(temp_pred))
  
  recall[t]<- sum(pred == 1 & actual == 1) / ( sum(pred == 0 & actual == 1) + sum(pred == 1 & actual == 1) )
  precision[t]<- sum(pred == 1 & actual == 1) / ( sum(pred == 1 & actual == 0) + sum(pred == 1 & actual == 1) )
  
  
  t<- t+1
  
  if(Thresh_hold[t] == 1)break 
}

table(pred, actual)
table(random_pred, actual)

plot(Thresh_hold, recall, type='l', lwd=3, ylim=c(0,1))
lines(Thresh_hold,precision)
which.max(recall + precision)
Thresh_hold[75] # 0.274

# 0.3 을 기준으로 제출 0.78468 별로 좋지 않다.

prob<- predict(fit_random_forest, titanic.test, type= 'prob')[,1]
pred<- prob < 0.3
submission$Survived<- as.factor(as.numeric(pred))

write.csv(submission, '20190106_3.csv', row.names = FALSE)
