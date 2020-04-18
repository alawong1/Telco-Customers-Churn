library(tidyverse)
library(psych) 
library(caret)
library(e1071)

library(rpart)
library(caTools)
library(dplyr)
library(party)
library(partykit)
library(rpart.plot)


setwd("D:/Other Code - D/Arrrrrr/-- Finance and Economics/telco_churn")

Telco <- read.csv("dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv")

head(Telco)

# Defintion: No = Didn't leave
#            Yes = Did leave
churnCount <- table(Telco$Churn)

bp <- barplot(churnCount, main="Distribution of Customer Churns", 
        xlab="Churns", ylab="Count", col=c("Blue", "Green"), ylim=c(0, 6000))
abline(h=0)
text(bp, (churnCount+200), labels=round(churnCount, digits=2))

# Creates bar chart of Churn against Partners
ggplot(Telco, mapping = aes(x = Partner))+
  geom_bar(stat = "count", aes(fill = Partner)) +
  facet_wrap(~Churn) +
  labs(title = "Bar Chart on Churn based on Partner of the Company",
       x = "Partner",
       y = "Number of People") +
  theme_light()

# Repeat the same bar chart but for Gender
ggplot(Telco, mapping = aes(x = gender))+
  geom_bar(stat = "count", aes(fill = gender)) +
  facet_wrap(~Churn) +
  labs(title = "Bar Chart on Churn based on Gender",
       x = "Gender",
       y = "Number of People") +
  theme_light()

Telco$gender = factor(Telco$gender)
Telco$Parter = factor(Telco$Partner)

Telco$customerID <- NULL

# Logistic Regression Model of the Telco Churn
Telco2 = Telco
Telco = rbind(sample_n(filter(Telco2, Churn=="Yes"), 1869), sample_n(filter(Telco, Churn=="No"), 1869))

smp_size <- floor(0.75 * nrow(Telco))

train_ind <- sample(seq_len(nrow(Telco)), size = smp_size)

train <- Telco[train_ind, ]
test <- Telco[-train_ind, ]

ChurnLogit = glm(Churn ~ .,
                 data=train,
                 family="binomial"
                 )

summary(ChurnLogit)

#creates a new column called EstimatedProb in test
resultTest = test %>% 
  mutate(EstimatedProb = predict(ChurnLogit,
                                 newdata = test,
                                 type = "response"))


# Now let's predict Y = 1 if P(Y = 1) > 0.6
resultTest1 = resultTest %>% mutate(predicted = I(EstimatedProb > 0.6) %>% as.numeric())

conf_matrix <- table(resultTest1$predicted, test$Churn)
conf_matrix

# Confusion Matrix

#     No  Yes
#  No 368 134
# Yes 100 333

# Decision Tree

rtree <- rpart(Churn ~., data=train)
rpart.plot(rtree)

