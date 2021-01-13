set.seed(1234567890)
library(geosphere)

stations <- read.csv("stations.csv", fileEncoding = "latin1")
temps <- read.csv("temps50k.csv")
st <- merge(stations, temps, by="station_number")
h_distance <- 50*10000 # These three values are up to the students
h_date <- 30
h_time <- 6
a <- 58.4274 # The point to predict (up to the students)
b <- 14.826
date <- "2013-11-04" # The date to predict (up to the students)
times <- c("04:00:00", "06:00:00", "08:00:00", "10:00:00", "12:00:00", 
           "14:00:00", "16:00:00", "18:00:00", "20:00:00", "22:00:00",
           "00:00:00")
temp <- vector(length=length(times))

gaussian_func <- function(distance) {
  return(exp(-(distance)^2))
}

# Geograthic distance
kernel1 <- function(x1, x2) {
  return(gaussian_func(distHaversine(c(as.numeric(x1["longitude"]), as.numeric(x1["latitude"])), 
                                     c(as.numeric(x2["longitude"]), as.numeric(x2["latitude"])))/h_distance))
}

# Days
kernel2 <- function(x1, x2) {
  return(gaussian_func(as.numeric(difftime(format(as.Date(x1["date"], "%Y-%m-%d"), "0000-%m-%d"), format(as.Date(x2["date"], "%Y-%m-%d"), "0000-%m-%d"), units="days"))))
}

# Time
kernel3 <- function(x1, x2) {
  return(gaussian_func(as.numeric(difftime(strptime(x1["time"], format="%H:%M:%S"), strptime(x2["time"], format="%H:%M:%S"), units = "hour")/h_time)))
}

sum_kernel <- function(x1, x2) {
  return(kernel1(x1, x2) + kernel2(x1, x2) + kernel3(x1, x2))
}

mult_kernel <- function(x1, x2) {
  return(kernel1(x1, x2) * kernel2(x1, x2) * kernel3(x1, x2))
}

kernel_pred <- function(x, data, kernel) {
  filtered_data = data[(as.Date(data$date) < x["date"]) | 
                         ((as.Date(data$date) == x["date"]) & (as.character(data$time) < x["time"])), ]
  result = apply(filtered_data, 1, function(x2) {return(kernel(x, x2))})
  return(filtered_data$air_temperature %*% result / sum(result))
}

forecast_points = data.frame(
  longitude = rep(b, length(times)),
  latitude = rep(a, length(times)),
  date = rep(as.Date(date), length(times)),
  time = times
)


temp_sum_kernel = apply(forecast_points, 1, function(x) {kernel_pred(x, st, sum_kernel)})
temp_mult_kernel = apply(forecast_points, 1, function(x) {kernel_pred(x, st, mult_kernel)})


# Kernel 1 (minus long/lat)
geo_distances = seq(0, 10000*100, 10000)
kernel_geo_distances = sapply(geo_distances, function(x) {gaussian_func(x/h_distance)})
plot(geo_distances, kernel_geo_distances, type = "l", xlab = "geographic distance (m)", ylab = "weight", main = "Kernel 1: Geographic Distance")

# Kernel 2
date_distances = seq(0, 180, 1)
kernel_date_distances = sapply(date_distances, function(x) {gaussian_func(x/h_date)})
plot(date_distances, kernel_date_distances, type = "l", xlab = "days", ylab = "weight", main = "Kernel 2: Days")

# Kernel 3
time_distances = seq(0, 12, 0.5)
kernel_time_distances = sapply(time_distances, function(x) {gaussian_func(x/h_time)})
plot(time_distances, kernel_time_distances, type = "l", xlab = "hours", ylab = "weight", main = "Kernel 3: Hours")


# Plot sum and mult kernels 
plot(seq(4, 24, 2), temp_sum_kernel, type="o", col = "red", ylim = c(3, 7), xlab = "time of day (24-hour format)", ylab = "temperature (°C)", main = "Summation and Multiplication Estimations")
points(seq(4, 24, 2), temp_mult_kernel, type="o", col = "blue")
legend("topleft", legend=c("Summation estimation", "Multiplication estimation"), col=c("red", "blue"), lty=1)





