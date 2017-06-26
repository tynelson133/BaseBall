Base <- read.csv("BASEBALL.csv")
#### Packages used
#install.packages("ggplot2")
library(ggplot2)
#install.packages("dplyr")
library(dplyr)
#install.packages("tidyr")
library(tidyr)
#install.packages("lazyeval")
library(lazyeval)
#install.packages("caret")
library(caret)
#install.packages("lme4")
library(lme4)
#install.packages("xgboost")
library(xgboost)
#install.packages("reshape2")
library(reshape2)


########## Cleaning the data:
# 1. Take any data point where the price was missing
# 2. Do not include a price over $300.00
#    only 1.8% of tickets are greater than $300.00 
# 3. create new variable 'NewSection' because there are sections that are not
#    numbered that are suites and are much more expensive.
# 4. There was one opponent name in which it had extra white space so I got rid of this
# 5. I created three new variables for year, month, and day
# 6. I ordered the day and month variables so they are more interpretable 
# 7. I removed the Month of October because this is when the championship was
#    and the prices are much higher here. If I wanted to just project what a 
#    playoff or championship ticket was I could use this information. 
BaseDat <- Base %>% filter(!is.na(Price), Price <= 300) %>%
  mutate(NewOpponent = trimws(Opponent, which = c('right'))) %>%
  select(-Opponent) %>% rename(Opponent = NewOpponent) %>%
  mutate(NewSection = as.character(Section)) %>%
  mutate(NewSection = ifelse(NewSection %in% 1:1000, 'General', 'Special')) %>%
  separate(Event.Date, c("Year","Month","Day"), "/", remove = FALSE) %>%
  mutate(Row = as.numeric(Row), 
         DayWeek = factor(weekdays(as.Date(Event.Date)),
                          levels = c('Monday', 'Tuesday',
                                     'Wednesday', 'Thursday',
                                     'Friday', 'Saturday',
                                     'Sunday')),
         MonthName = factor(months(as.Date(Event.Date)),
                            levels = c('April', 'May', 'June', 'July',
                                       'August', 'September', 'October'))) %>%
  filter(MonthName != 'October') 

###############################################################################
################# Summary Tables of Prices by Variable ########################
###############################################################################
##### A function for making summary tables 
SummaryFunc <- function(Var, Price){
  data.frame(BaseDat %>% group_by_(Var) %>% 
             summarize('Mean' = mean(Price), 'Median' = median(Price),
                       'SD' = sd(Price), 
                       'Minimum' = min(Price), 
                       'Q1' = quantile(Price, probs = 0.25),
                       'Q3' = quantile(Price, probs = 0.75),
                       'Maximum' = max(Price)))
}

### Each summary Table
OpponentTable <- SummaryFunc(Var = 'Opponent')
SectionTable  <- SummaryFunc(Var = 'NewSection')
DayTable      <- SummaryFunc(Var = 'DayWeek')
MonthTable    <- SummaryFunc(Var = 'MonthName')
AreaTable     <- SummaryFunc(Var = 'Area')


###############################################################################
####################### Visualizing Data ######################################
###############################################################################
######### Making BoxPlots: I only show one but the same can be done for all
ggplot(BaseDat, aes(DayWeek, Price)) + geom_boxplot() + coord_flip()

########### A function to create bar plots for various different 
# variables for the median ticket price. 
# Would like to scale this to a Shiny app or learn how to create a dashboard
BarFunc <- function(Data, X, Title, xlab = 'X', Angle = 50){
  ### Creating data to use for barplot
  NewDat <- Data %>% group_by_(X) %>% 
    summarise(Median = median(Price))
  ### Plotting barplot
  ggplot(NewDat, aes_string(x = X, y = 'Median')) +
    geom_bar(stat = "identity") + 
    labs(x = xlab, y = "Median Ticket Price", title = Title) +
    theme(axis.text.x = element_text(angle = Angle, hjust = 1))
}
#### Plotting Median ticket prices by Month
MonthPlot <- BarFunc(Data = BaseDat, X = "MonthName", 
        Title = "Median Ticket Price by Month",
        xlab = "Month")
#### Plotting Median ticket prices by Day
DayPlot <- BarFunc(Data = BaseDat, X = "DayWeek", 
                     Title = "Median Ticket Price by Day",
                     xlab = "Day")
#### Plotting Median ticket prices by Weather Status
WeatherPlot <- BarFunc(Data = BaseDat, X = "Weather", 
                    Title = "Median Ticket Price by Weather",
                    xlab = "Weather")

#### Plotting Median ticket prices by Weather Status
AreaPlot <- BarFunc(Data = BaseDat, X = "Area", 
                       Title = "Median Ticket Price by Area",
                       xlab = "Area", Angle = 0)

###### Plotting multiple plots on one page
load("MultiPlotFunction.Rdata")

#### Saving File
pdf(file = "MedianSummaryPlots.pdf")
SummaryPlot <- multiplot(MonthPlot, DayPlot, WeatherPlot, AreaPlot, cols = 2)
dev.off()

#### Plotting Median ticket prices by Day
TeamPlot <- BarFunc(Data = BaseDat, X = "Opponent", 
                    Title = "Median Ticket Price by Opponent",
                    xlab = "Opponent")
#### Saving TeamPlot
ggsave("MedianTeamPlot.pdf", plot = TeamPlot)


########## This is used to construct a dataframe for number of tickets and 
######### average ticket sales by each game
TicketByDay <- BaseDat %>% group_by(MonthName, Day, Opponent, Weather) %>% 
  summarise(Num_tickets = length(Event.Date), MedSales = median(Price)) %>% 
  arrange(MonthName, Day) 
### Creating a game number variable and adding it to the dataset above
TicketByDay$GameNumber <- seq_along(TicketByDay$Day)

### Plotting Sales of tickets by Game.
ScatterPrices <- ggplot(TicketByDay, aes(x = GameNumber, y = MedSales, color = Num_tickets)) +
  geom_point() +
  labs(x = "Game Number", y = "Median Price of Ticket", color = "Number of Tickets")
### Saving scatter plot
ggsave("MedianPriceGame.pdf", plot = ScatterPrices)

########################################################################
####### Developing a Model to predict regular season Prices ############
########################################################################
####### Constructed a crossvalidation test for the following models:
# 1. Linear Model
# 2. GLM with gamma ditribution and log link
# 3. Log Linear Model
# 4. xgboost model

####### Compared all the models based on the mean squared error

# Naming MSE vector to be saved from loop
# If I was to use this model for actual prediction I would want to save
# Each of the coefficients so that I could use mean or median value for 
# the final model
MSEVec <- list()

### Using 10 fold cross-validation 
Folds <- createFolds(BaseDat$Price, k = 10, 
                     list = TRUE, 
                     returnTrain = FALSE)

################ Crossvalidation For loop #####################
for(i in 1:length(Folds)){

### Train and Test data 
TrainDat <- BaseDat[-Folds[[i]], ]
TestDat  <- BaseDat[Folds[[i]], ]

### Linear Model
LMMod <- glm(Price ~ Row + Area + Weather + Opponent + NewSection, data = TrainDat,
             family = gaussian)
### Log linear model
LLMMod <- glm(log(Price) ~ Row + Area + Weather + Opponent + NewSection, data = TrainDat,
              family = gaussian)
### generalized linear model with gamma disribution. (Very similiar to log linear model)
GLMMod <- glm(Price ~ Row + Area + Weather + Opponent + NewSection, data = TrainDat,
           family = Gamma(link = "log"))

######################## XGBoost Model ###########################
## Data needs to be characterized a little different to use XGBoost

### Train data in the format needed for xgboost
XGTrainDat <- TrainDat %>% select(Row, Area, Weather, Opponent, NewSection)
XGTrain <- model.matrix(~. -1, data = XGTrainDat, sparse = TRUE)

###### The Outcome of the trained model
OutcomeTrain <- TrainDat$Price

### XGBoost model
## Note that I have already run a analysis to check parameter values in XGBoost
bst2 <- xgboost(data = XGTrain, label = OutcomeTrain, max_depth = 5,
                eval_metric = 'rmse', 
                eta = .2, nthread = 2, nrounds = 300, 
                objective = "reg:gamma")

###################### Predicting the test data #########################
XGTestDat <- TestDat %>% select(Row, Area, Weather, Opponent)
XGTest <- model.matrix(~. -1, data = XGTestDat, sparse = TRUE)


### The predicted values for the test data for each  of the four models
LMPred  <- predict(LMMod,newdata = TestDat ,type = 'response')
LLMPred <- exp(predict(LLMMod,newdata = TestDat ,type = 'response'))
GLMPred <- predict(GLMMod, newdata = TestDat, type = 'response')
XGPred  <- predict(bst2, XGTest)
### Actual Prices
TestPrice <- TestDat$Price 

### Creating a dataframe of the models and the predicted prices
AllPred <- melt(data.frame(TestPrice, LMPred, LLMPred, GLMPred, XGPred))
colnames(AllPred) <- c('Model', 'Price')

### Linear Plot
LinearPlot <- ggplot(data=subset(AllPred, Model == c('TestPrice', 'LMPred')), 
                 aes(x = Price, fill = Model)) + 
  geom_histogram(alpha = 0.4, position = 'identity') + xlim(0,320)
### Log Linear Plot
LogLinearPlot <- ggplot(data=subset(AllPred, Model == c('TestPrice', 'LLMPred')), 
                 aes(x = Price, fill = Model)) + 
  geom_histogram(alpha = 0.4, position = 'identity') + xlim(0,320)
### Generalized Gamma Plot
GLMPlot <- ggplot(data=subset(AllPred, Model == c('TestPrice', 'GLMPred')), 
                 aes(x = Price, fill = Model)) + 
  geom_histogram(alpha = 0.4, position = 'identity') + xlim(0,320)
### XGBoost Plot
XGPlot <- ggplot(data=subset(AllPred, Model == c('TestPrice', 'XGPred')), 
       aes(x = Price, fill = Model)) + 
  geom_histogram(alpha = 0.4, position = 'identity') + xlim(0,320)

### Putting all the plots together for each fold and saving File
pdf(file = paste("HistFold", i, ".pdf", sep = ""))
AllPlot <- multiplot(LinearPlot, LogLinearPlot, GLMPlot, XGPlot, cols = 2)
dev.off()

### RMSE for each of the models
LMMSE  <- sqrt(sum((TestDat$Price - LMPred)^2)/dim(TestDat)[1])
LLMMSE <- sqrt(sum((TestDat$Price - LLMPred)^2)/dim(TestDat)[1])
GLMMSE <- sqrt(sum((TestDat$Price - GLMPred)^2)/dim(TestDat)[1])
XGMSE  <- sqrt(sum((TestDat$Price - XGPred)^2)/dim(TestDat)[1])

MSEVec[[i]] <- c('Linear MSE' = LMMSE, 
                 'Log Linear MSE' = LLMMSE, 
                 'Generalized MSE' = GLMMSE, 
                 'XGBoost MSE' = XGMSE)
}

MSEFolds <- do.call(rbind, MSEVec)

# BaseBall
