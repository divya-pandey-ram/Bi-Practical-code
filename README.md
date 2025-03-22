# Bi-Practical-code
**Time Series**
library(ggplot2)
library(forecast)

data("AirPassengers")
ts_data <- ts(AirPassengers, start = c(1949, 1), frequency = 12)


ts_decomposed <- decompose(ts_data, type = "multiplicative")

par(mfrow = c(3,1))
plot(ts_decomposed$trend)
plot(ts_decomposed$seasonal) 
plot(ts_decomposed$random)
par(mfrow = c(1,1))

# ARIMA Model
arima_model <- arima(ts_data, order = c(2,2,1))

forecast_value <- predict(arima_model, n.ahead = 12)

plot(ts_data, xlim = c(1949, 1962), main = "Air Passenger Flow",
     xlab = "Year", ylab = "Passengers in Thousands", col = "blue", lwd = 2)


forecast_years <- seq(1961, 1962, length.out = 12)
lines(forecast_years, forecast_value$pred, col = "red", lwd = 2)
points(forecast_years, forecast_value$pred, col = "red", pch = 19)

**classification**
# KNN Method

bankloan <- read.csv("~/Desktop/BI Prac/BANK LOAN.csv")

library(caret)
index <- createDataPartition(bankloan$DEFAULTER, p=0.8, list=FALSE)
View(bankloan)

bankloan2 <- subset(bankloan, select = c(-DEFAULTER))
bankloan3 <- scale(bankloan2)

traindata <- bankloan3[index,]
testdata <- bankloan2[-index,]

#create class vector
Ytrain <- bankloan$DEFAULTER[index]
Ytest <- bankloan$DEFAULTER[-index]

library(class)
model_knn <- knn(traindata, testdata, k=23, cl= Ytrain)
table(Ytest, model_knn)

specificity(as.factor(Ytest),as.factor(model_knn),cutoff = 0.5)         
sensitivity(as.factor(Ytest),as.factor(model_knn),cutoff = 0.5).

**Decesion Tree**
library(caret)
library(ggplot2)
library(rpart)
library(rpart.plot)
library(e1071)

data <- read.csv("C:\\Users\\Divya\\OneDrive\\Desktop\\BANK LOAN.CSV")
head(data)
data$default <- as.factor(data$default)
str(data)

set.seed(42)
splitIndex <- createDataPartition(data$default, p = 0.8, list = FALSE)
traindata <- data[splitIndex, ]
testdata <- data[splitIndex, ]

#checking table distribution
table(traindata$default)
table(testdata$default)

#Train Decision Tree Model
dt_model <- rpart(default ~ ., data = data, method = "class") #by default it is Gini index
print(dt_model)
png("Decision Tree.png", width = 800, height = 600)
rpart.plot(dt_model,
           main = "Decision Tree Chart",
           cex = 1)
dev.off()

#Confusion Matrix
dt_prob <- predict(dt_model, testdata, type = "prob")
dt_pred <- ifelse(dt_prob[,2] > 0.5, 1, 0)
dt_pred <- as.factor(dt_pred)

dt_conf_matrix <- confusionMatrix(dt_pred, testdata$default)
dt_conf_matrix

#ROC Curve
library(pROC)
dt_roc_curve <- roc(testdata$default, dt_prob[,2])
plot(dt_roc_curve,
     main = "ROC Curve - Decision Tree",
     col = "purple",
     lwd = 2)
dt_auc_roc <- auc(dt_roc_curve)
cat("Decision Tree AUC: ", dt_auc_roc)

**Kmeans**

library(ggplot2)
data("iris")
iris_data <- iris[, -5]

set.seed(58)

kmeans_result <- kmeans(iris_data, centers = 3, nstart = 25)
print(kmeans_result)

iris$cluster <- as.factor(kmeans_result$cluster)

table(iris$Species, iris$cluster)

ggplot(iris, aes(x = Sepal.Length, y = Sepal.Width)) + 
  geom_point(size = 3) + 
  ggtitle("K-means Clustering") + 
  theme_minimal()

**Linear Regression**

# LINEAR REGRESSION
library(ggplot2)
library(caret)

# load inbuilt data
data("mtcars")
set.seed(89)

cor(mtcars) # co-relation matrix for the data

sample_index <- sample(1:nrow(mtcars), 0.8 * nrow(mtcars))
train_data <- mtcars[sample_index, ]
test_data <- mtcars[-sample_index, ]

linearModel <- lm(mpg ~ hp + drat + am, data = train_data)
summary(linearModel)
