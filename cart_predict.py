import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

# Load the dataset
data = pd.read_csv("C:/Users/furkan/Desktop/Yapay zeka/Projeler/Credic-Cart_predcit/cart-perdict.csv")

# Explore the dataset
print("First 5 observations of the dataset:\n", data.head())
print("\nInformation about the dataset:\n", data.info())
print("\nLast 5 observations of the dataset:\n", data.tail())

# Identify missing data
print("\nNumber of missing values in each column:\n", data.isnull().sum())
print("\nTotal number of missing values:\n", data.isnull().sum().sum())

# Replace '?' values with NaN
cc_apps = data.replace('?', np.nan)

# Check for NaN values
print("\nNumber of NaN values in each column:\n", cc_apps.isnull().sum())

# Fill NaN values for object values
for col in cc_apps:
    if cc_apps[col].dtypes == 'object':
        cc_apps = cc_apps.fillna(cc_apps[col].value_counts().index[0])

# Check for NaN values
print("\nNumber of NaN values in each column (after filling):\n", cc_apps.isnull().sum())

# Label Encode object values
le = LabelEncoder()

for col in cc_apps.columns.to_numpy():
    if cc_apps[col].dtypes == 'object':
        cc_apps[col] = le.fit_transform(cc_apps[col])

# Convert categorical variables into numerical ones using One-Hot Encoding
cc_apps = pd.get_dummies(cc_apps)

# Split the data into training and test sets
cc_apps = cc_apps.drop([11, 13], axis=0)
cc_apps = cc_apps.to_numpy()
X = cc_apps[:, :-1]
y = cc_apps[:, -1:]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Create Normalizer for X and y
scaler = MinMaxScaler(feature_range=(0, 1))
re_X_train = scaler.fit_transform(X_train)
re_X_test = scaler.transform(X_test)

# Create StandardScaler for X and y
scaler = StandardScaler()
re_X_train = scaler.fit_transform(X_train)
re_y_train = scaler.fit_transform(y_train)

# Logistic Regression model instantiation and training
logreg = LogisticRegression(random_state=42)
logreg.fit(re_X_train, y_train)

# Predict with the Logistic Regression model
pred_lr = logreg.predict(re_X_test)

# Calculate accuracy score of Logistic Regression model
accuracy_score_lr = accuracy_score(y_test, pred_lr)
print("\nAccuracy Score (Logistic Regression): ", round(accuracy_score_lr, 2))

# Decision Tree model instantiation and training
tree = DecisionTreeClassifier()
tree.fit(re_X_train, y_train)

# Predict with the Decision Tree model
pred_tr = tree.predict(re_X_test)

# Calculate accuracy score of Decision Tree model
accuracy_score_tr = accuracy_score(y_test, pred_tr)
print("\nAccuracy Score (Decision Tree): ", round(accuracy_score_tr, 2))

# Random Forest model instantiation and training
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(re_X_train, y_train)

# Predict with the Random Forest model
pred_rf = rf.predict(re_X_test)

# Calculate accuracy score of Random Forest model
accuracy_score_rf = accuracy_score(y_test, pred_rf)
print("\nAccuracy Score (Random Forest): ", round(accuracy_score_rf, 2))

# Gradient Boosting model instantiation and training
gbc = GradientBoostingClassifier(random_state=42)
gbc.fit(re_X_train, y_train)

# Predict with the Gradient Boosting model
pred_gbc = gbc.predict(re_X_test)

# Calculate accuracy score of Gradient Boosting model
accuracy_score_gbc = accuracy_score(y_test, pred_gbc)
print("\nAccuracy Score (Gradient Boosting): ", round(accuracy_score_gbc, 2))

# KNN model instantiation and training
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(re_X_train, y_train)

# Predict with the KNN model
pred_knn = knn.predict(re_X_test)

# Calculate accuracy score of KNN model
accuracy_score_knn = accuracy_score(y_test, pred_knn)
print("\nAccuracy Score (KNN): ", round(accuracy_score_knn, 2))

# Visualize the Confusion Matrix for Decision Trees
y_pred = tree.predict(re_X_test)
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10, 10))
sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix for Decision Trees")
plt.show()

# Visualize the Confusion Matrix for Logistic Regression
y_pred_log = logreg.predict(re_X_test)
cm_log = confusion_matrix(y_test, y_pred_log)

plt.figure(figsize=(8, 6))
sns.heatmap(cm_log, annot=True, fmt="d", cmap="YlGnBu")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix for Logistic Regression")
plt.show()
