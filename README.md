# Bi-Practical-code
*classification*
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
