getwd()
setwd("C:\\Users\\My guest\\Desktop\\data science\\projectcab")
cab<-read.csv("train_cab.csv",stringsAsFactors = FALSE)
str(cab)
cab$fare_amount<-as.numeric(cab$fare_amount)
cab$pickup_datetime<-as.POSIXct(cab$pickup_datetime,format="%Y-%m-%d %H:%M:%S")

#convert all zero values to NA
#cab[cab==0]<-NA
sum(is.na(cab))


#***********************************************************************************************************************************************************************************
#FINDING MISSING VALUE AND CONVERTING INTO DATA FRAME
#************************************************************************************************************************************************************************************
missing_val<-data.frame(apply(cab,2,function(x){sum(is.na(x))}))
missing_val

#convert row names into columns
missing_val$columns=row.names(missing_val)
row.names(missing_val)<-NULL
#remane first variable name 
names(missing_val)[1]<-"missing values"
#calculate the percent of missing values in each variable and saving it in missing_val
missing_val$missing_val_percent<-(missing_val$`missing values`/nrow(cab)) * 100
missing_val
#arranging the missing values in decending values
missing_val<-missing_val[order(-missing_val$missing_val_percent),]

#rearanging columns
missing_val<-missing_val[,c(2,3)]

missing_val
#write output results back into harddisk
#write.csv(missing_val,"missing_perc.csv",row.names = F)
#write.csv(cab,"cab.csv",row.names = F)
View(cab)


#*************************************************************************************************************************************************************************************
#MISSING VALUE ANALYSIS IN FARE_AMOUNT(FISRT VARIABLE)
#*************************************************************************************************************************************************************************************

#actual value cab[81,1]=5.7
#mean=15.01
#median=8.5
#knn=14.74

cab[81,1]
cab[81,1]=NA
mean(cab$fare_amount,na.rm = T)
#meadian=8.5
median(cab$fare_amount,na.rm = T)
 
install.packages("DMwR2")
library(DMwR2)
cabk<-knnImputation(cab,k=5)
cabk[81,1] 
#knn=8

#since knn value is closest we choose imputation usin knn method
#cab$fare_amount[is.na(cab$fare_amount)]=mean(cab$fare_amount,na.rm = T)
cab$fare_amount=cabk$fare_amount
cab[81,1]
#check if any missing value left
sum(is.na(cab$fare_amount))
sum(is.na(cab))

#*************************************************************************************************************************************************************************************
#MISSING VALUE ANALYSIS IN SECOND VARIABLE:picup_datetime
#*************************************************************************************************************************************************************************************
install.packages("zoo")
library(zoo)
cab$pickup_datetime=na.locf(na.locf(cab$pickup_datetime), fromLast = TRUE)

#check if any missing value left
sum(is.na(cab$pickup_datetime))
sum(is.na(cab))

#*************************************************************************************************************************************************************************************
#MISSING VALUE ANALYSIS IN THIRD VARIABLE:PICKUP_LONGITUDE
#*************************************************************************************************************************************************************************************
#check if any missing value ARE PRESENT
sum(is.na(cab$pickup_longitude))

#*********************************************************************************************************************************************************************************
#MISSING VALUE ANALYSIS IN FOURTH VARIABLE:PICKUP_LATITUDE
#*************************************************************************************************************************************************************************************

#check if any missing value ARE PRESENT
sum(is.na(cab$pickup_latitude))

#*************************************************************************************************************************************************************************************
#MISSING VALUE ANALYSIS IN FIFTH VARIABLE: dropoff_longitude
#*************************************************************************************************************************************************************************************

#check if any missing value ARE PRESENT
sum(is.na(cab$dropoff_longitude))

#*************************************************************************************************************************************************************************************
#MISSING VALUE ANALYSIS IN SIXTH VARIABLE: dropoff_latitude
#*************************************************************************************************************************************************************************************

#check if any missing value ARE PRESENT
sum(is.na(cab$dropoff_latitude))

 
#*************************************************************************************************************************************************************************************
#MISSING VALUE ANALYSIS IN SEVEN VARIABLE: passenger_count 
#*************************************************************************************************************************************************************************************

#check if any missing value ARE PRESENT
sum(is.na(cab$passenger_count))

#actual value 
#cab[81,7]= 1
#mean= 2.634551
#median= 1
#knn= 3.600001

cab[81,7]
cab[81,7]=NA
mean(cab$passenger_count,na.rm = T)
#meadian
median(cab$passenger_count,na.rm = T)
 
install.packages("DMwR2")
library(DMwR2)
cabk<-knnImputation(cab,k=5)
cabk[81,7] 
#knn=2.2
#since median value is closest we choose imputation usin median method
cab$passenger_count[is.na(cab$passenger_count)]=median(cab$passenger_count,na.rm = T)
cab[81,7]
#check if any missing value left
sum(is.na(cab$passenger_count))
sum(is.na(cab))
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#FINALLY DATA IS FREE FROM ALL MISSING VALUES
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@


#**************************************************************************************
#OUTLIER ANALYSIS
#**************************************************************************************
df<-cab
#outlier analysis
#selecting all numeric variables becaues outliers work on numeric variables only
numeric_index<-sapply(cab,is.numeric)
numeric_index
numeric_data<-cab[,numeric_index]
cnames<-colnames(numeric_data)
cnames

#ploting box plot
par(mfrow=c(3,2))
for (i in cnames)  {
  boxplot(cab[i],main="Boxplot",ylab=i,ylim=c(),las=1)
}

#removing outliesr
for (i in cnames) {
  print(i)
  val<-cab[,i][cab[,i]%in%boxplot.stats(cab[,i])$out]
  print(length(val))
  cab<-cab[which(!cab[,i] %in% val),]
}

str(cab)
summary(cab$fare_amount)
summary(cab$passenger_count)
length(cab$passenger_count)
#boxplot after outlier removal
boxplot(df[[1]])

write.csv(cab,"cab_df.csv",row.names = F)

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# test data  is free from missing values but has outliers
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

#**************************************************************************************
#OUTLIER ANALYSIS in test.csv

#import data in r
test_cab<-read.csv("test.csv",stringsAsFactors = FALSE)
test_cab$pickup_datetime<-as.POSIXct(test_cab$pickup_datetime,format="%Y-%m-%d %H:%M:%S")
dff<-test_cab
#outlier analysis
#selecting all numeric variables becaues outliers work on numeric variables only
numeric_index<-sapply(test_cab,is.numeric)
numeric_index
numeric_data<-test_cab[,numeric_index]
cnames<-colnames(numeric_data)
cnames

#ploting box plot
par(mfrow=c(3,2))
for (i in cnames)  {
  boxplot(test_cab[i],main="Boxplot",ylab=i,ylim=c(),las=1)
}

#removing outliesr
for (i in cnames) {
  print(i)
  val<-test_cab[,i][test_cab[,i]%in%boxplot.stats(test_cab[,i])$out]
  print(length(val))
  test_cab<-test_cab[which(!test_cab[,i] %in% val),]
}

str(test_cab)
summary(test_cab$passenger_count)
length(test_cab$passenger_count)
#boxplot after outlier removal
boxplot(test_cab[[1]])

write.csv(test_cab,"test_cab_df.csv",row.names = F)

#**************************************************************************************
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#data is free from outliesrs
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

#************************************************************************************************************************
#feature selection in both train and test data
#*********************************************************************************************************************
 #using correlation plot becaues we ahe continous variable 
library(corrgram)

#for train data
corrgram(cab[,numeric_index],order = FALSE,
         upper.panel = panel.pie,text.panel = panel.txt,main="correlation plot")

# for test data
corrgram(test_cab[,numeric_index],order = FALSE,
         upper.panel = panel.pie,text.panel = panel.txt,main="correlation plot")


#instead of removing we will do feature engineering  now
#we will extract manhatton distance from coordinates and months and weekdays from date variables
install.packages("tidyverse")
install.packages("lubridate")
install.packages("devtools")
library(tidyverse)
library(lubridate)
library(devtools)
cab$years<-year(cab$pickup_datetime)
cab$months<-months(cab$pickup_datetime)
cab$weekdays<-weekdays(cab$pickup_datetime)
class(cab$months)
class(cab$weekdays)
summary(cab$years)
head(cab)
summary(cab)
str(cab)

# for test data
#instead of removing we will do feature engineering  now
#we will extract manhatton distance from coordinates and months and weekdays from date variables
install.packages("tidyverse")
install.packages("lubridate")
install.packages("devtools")
library(tidyverse)
library(lubridate)
library(devtools)
Test_cab$years<-year(cab$pickup_datetime)
test_cab$months<-months(cab$pickup_datetime)
test_cab$weekdays<-weekdays(cab$pickup_datetime)
class(test_cab$months)
class(test_cab$weekdays)
summary(test_cab$years)
head(test_cab)
summary(test_cab)
str(test_cab)

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#feature engineering
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# for train data
diff_longi<-abs(cab$pickup_longitude-cab$dropoff_longitude)
diff_latit<-abs(cab$pickup_latitude-cab$dropoff_latitude)
cab$distance<-diff_latit+diff_longi
head(cab)
cab$months<-as.factor(cab$months)
cab$months<-as.integer(cab$months)
cab$weekdays<-as.factor(cab$weekdays)
cab$weekdays<-as.integer(cab$weekdays)
summary(cab$distance)
newcab<-subset(cab,select = -c(pickup_longitude,dropoff_longitude,pickup_latitude,dropoff_latitude,pickup_datetime))
sum(is.na(newcab))

numeric_index<-sapply(newcab,is.numeric)
numeric_index
corrgram(newcab[,numeric_index],order = FALSE,
               upper.panel = panel.pie,text.panel = panel.txt,main="correlation plot")
head(newcab)

corrgram(cab[,numeric_index],order = FALSE,
         upper.panel = panel.pie,text.panel = panel.txt,main="correlation plot")


# test data 
diff_longi<-abs(test_cab$pickup_longitude-test_cab$dropoff_longitude)
diff_latit<-abs(test_cab$pickup_latitude-test_cab$dropoff_latitude)
test_cab$distance<-diff_latit+diff_longi
head(test_cab)
test_cab$months<-as.factor(test_cab$months)
test_cab$months<-as.integer(test_cab$months)
test_cab$weekdays<-as.factor(test_cab$weekdays)
test_cab$weekdays<-as.integer(test_cab$weekdays)
summary(test_cab$distance)
newcab_test<-subset(test_cab,select = -c(pickup_longitude,dropoff_longitude,pickup_latitude,dropoff_latitude,pickup_datetime))
sum(is.na(newcab_test))

numeric_index<-sapply(newcab_test,is.numeric)
numeric_index
corrgram(newcab_test[,numeric_index],order = FALSE,
         upper.panel = panel.pie,text.panel = panel.txt,main="correlation plot")
head(newcab_test)

corrgram(test_cab[,numeric_index],order = FALSE,
         upper.panel = panel.pie,text.panel = panel.txt,main="correlation plot")

#train data is ready to build model upon them
write.csv(newcab,"newcab.csv",row.names = F)
#test data is ready to build model upon them
write.csv(newcab_test,"newcab_test.csv",row.names = F)

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#linear regression model
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

dim(newcab)
#checking multicolinearity
install.packages("usdm")
library(usdm)
vif(newcab[,-1])
vifcor(newcab[,-1],th=0.9)
library(rpart)
library(MASS)
df1=newcab
head(df1)
#divide the data into train and test
train_index = sample(1:nrow(df1),0.8*nrow(df1))
train=df1[train_index,]
test=df1[-train_index,]

#bulit model
lm_model=lm(fare_amount~.,data = train)
#summary
summary(lm_model)

#PREDICTING VALUES
prediction_lr=predict(lm_model,test[,2:6])
#test[,2:6]

#check accuracy
mape=function(y,yhat){
  mean(abs((y-yhat)/y))
}
mape(test[,1],prediction_lr)  
#0.2012that means 80% accruracy

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#DECISION TREE
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
library(rpart)
library(MASS)
df1=newcab
head(df1)
#divide the data into train and test
train_index = sample(1:nrow(df1),0.8*nrow(df1))
train=df1[train_index,]
test=df1[-train_index,]


#r part for regression
fit_dt=rpart(fare_amount~.,data=train,method = "anova")

#prediction for new test case
predictions_dt=predict(fit_dt,test[,-1])

# checking accuracy using mape
mape=function(y,yhat){
    mean(abs((y-yhat)/y))
  }
mape(test[,1],predictions_dt)  
#0.222
#model is 78% accurate

#alternative method
library(DMwR)
regr.eval(test[,1],predictions_dt,stats=c('mae','rmse','mape','mse'))

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#random forest
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
install.packages("randomForest")
library(randomForest)

#divide the data into train and test
train_index = sample(1:nrow(df1),0.8*nrow(df1))
train=df1[train_index,]
test=df1[-train_index,]

rf_model=randomForest(fare_amount~.,train,importance=TRUE,ntree=100)

install.packages("inTrees")
library(inTrees)
treeList=RF2List(rf_model)

#extract rules
exec=extractRules(treeList,train[,-1])

#visualise
exec[1:2,]
#make rules more readable
readableRules = presentRules(exec,colnames(train))
readableRules[1:2,]

#get rule matrics
ruleMetric=getRuleMetric(exec,train[,-1],train$fare_amount)

#evaluate few rule matric
ruleMetric[1:2,]

#predict test data useng random forest model
prediction_rf=predict(rf_model,test[,-1])

# checking accuracy using mape
mape=function(y,yhat){
  mean(abs((y-yhat)/y))
}
mape(test[,1],prediction_rf)  
# .1488
#model has 86% percent accuracy


#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#model selection
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#so finally we have got86 % of accuracy for random forestt
#so we will implement it with the test data

prediction_rf=predict(rf_model,newcab_test)
