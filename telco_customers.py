import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
sns.set(style="darkgrid")
import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

# What does Churn in this dataset mean?
# Customers who left within the last month

# No - didn't leave the service in the last month.
# Yes - did leave the service in the last month.
'''
1. customerID  - Customer ID
2. gender      - Whether the customer is a male or a female
3. SeniorCitizen - Whether the customer is a senior citizen or not 
                (1, 0)
4. Partner - Whether the customer has a partner or not 
                (Yes, No)
5. Dependents - Whether the customer has dependents or not 
                (Yes, No)
6. tenure - Number of months the customer has stayed with the company
7. PhoneService - Whether the customer has a phone service or not 
                (Yes, No)
8. MultipleLines - Whether the customer has multiple lines or not 
                (Yes, No, No phone service)
9. InternetService - Customer’s internet service provider 
                (DSL, Fiber optic, No)
10. OnlineSecurity - Whether the customer has online security or not 
                (Yes, No, No internet service)
11. OnlineBackup - Whether the customer has online backup or not 
                (Yes, No, No internet service)
12. DeviceProtection - Whether the customer has device protection or not 
                (Yes, No, No internet service)
13. TechSupport - Whether the customer has tech support or not 
                (Yes, No, No internet service)
14. StreamingTV - Whether the customer has streaming TV or not 
                (Yes, No, No internet service)
15. StreamingMovies - Whether the customer has streaming movies or not 
                (Yes, No, No internet service)
16. Contract - The contract term of the customer 
                (Month-to-month, One year, Two year)
17. PaperlessBilling - Whether the customer has paperless billing or not 
                (Yes, No)
18. PaymentMethod - The customer’s payment method 
                (Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic))
19. MonthlyCharges - The amount charged to the customer monthly
20. TotalCharges - The total amount charged to the customer

'''
df = pd.read_csv("dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv")
# print(df.shape)
# print(df.head())

# print(df.columns)

def count_plot(label):

    ax = sns.countplot(x=label, data=df)
    title_text = label[0].upper() + label[1:]
    # print(title_text)
    plt.title(f"Count of {title_text}")
    # Placing numbers on top of the bars.
    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x() + p.get_width()/2., height + 3, height, ha="center") 
    plt.show()

#count_plot("Churn")
# 1800
# count_plot("Partner")

# print(df.dtypes)

churn = df.loc[df["Churn"] == "Yes"]
no_churn = df.loc[df["Churn"] == "No"][:1869]

print(churn.shape)
print(no_churn.shape)

new_df = pd.concat([churn, no_churn])
#print(new_df.head())

for c in new_df.columns: 
    if new_df[c].dtypes == "object":
        lbl = LabelEncoder()
        lbl.fit(list(new_df[c].values))
        new_df[c] = lbl.transform(list(new_df[c].values))

#print(new_df.head())

X = new_df.drop(["customerID", "Churn"], axis=1)
y = new_df["Churn"]

X = (X - X.mean()) / X.std()

#print(X.head())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")

cv_iter = 100
validation_scores = []

# ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
log_reg = LogisticRegression(max_iter=3000, n_jobs=-1)
log_reg_score = cross_val_score(log_reg, X, y, scoring="roc_auc", cv=cv_iter)
validation_scores.append(log_reg_score)

print("Mean AUC Score - Logistic Regression: ", log_reg_score.mean())

# ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
dt_clf = DecisionTreeClassifier(random_state=42)
dt_score = cross_val_score(dt_clf, X, y, scoring="roc_auc", cv=cv_iter) 
validation_scores.append(dt_score)

print("Mean AUC Score - Decision Tree: ", dt_score.mean())

# ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
rf_clf = RandomForestClassifier(n_estimators=100, min_samples_split=3, max_leaf_nodes=16, n_jobs=-1)
rf_cv_score = cross_val_score(rf_clf, X, y, scoring="roc_auc", cv=cv_iter)
validation_scores.append(rf_cv_score)

print("Mean AUC Score - Random Forest: ", rf_cv_score.mean())

# ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
gb_clf = GradientBoostingClassifier(min_samples_leaf=4, min_samples_split=4, max_depth=9)
gb_score = cross_val_score(gb_clf, X, y, scoring="roc_auc", cv=cv_iter)
validation_scores.append(gb_score)

print("Mean AUC Score - Gradient Boost: ", gb_score.mean())

log_reg.fit(X_train, y_train)
rf_clf.fit(X_train, y_train)
dt_clf.fit(X_train, y_train)
gb_clf.fit(X_train, y_train)

print("-"*25)
# Get feature_importances of Random Forest for all features
rf_features = []
for name, score in zip(X, rf_clf.feature_importances_):
    rf_features.append([score, name])
    
rf_features = sorted(rf_features, reverse=True)
print("Feature importance of Random Forest")
for scores in rf_features:
    print("{:.7f} - {} ".format(scores[0], scores[1]))

print("-"*25)
# Get feature_importances of Decision Tree for all features
dt_features = []
for name, score in zip(X, dt_clf.feature_importances_):
    dt_features.append([score, name])
    
dt_features = sorted(dt_features, reverse=True)
print("Feature Importance of Decision Tree")
for scores in dt_features:
    print("{:.7f} - {} ".format(scores[0], scores[1]))   


algo_names = ["LogisticRegression", "DecisionTreeClassifier", "RandomForestClassifier", "GradientBoostingClassifier"]
def box_plot():
    fig = plt.figure(figsize=(9, 5))
    fig.suptitle("Comparison of different Algorithms", fontsize=22)
    ax = fig.add_subplot(111)
    sns.boxplot(x=algo_names, y=validation_scores)
    ax.set_xticklabels(algo_names)
    ax.set_xlabel("Algorithm", fontsize=14)
    ax.set_ylabel("Mean AUC Score", fontsize=18)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)   # Creates the x_ticks to be tilted on it's diagonal.
    plt.ylim((0.2, 1.0))
    plt.show()

box_plot()

