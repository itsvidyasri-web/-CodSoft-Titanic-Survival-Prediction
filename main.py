
import pandas as pd

# Load Titanic dataset
data = pd.read_csv("Titanic-Dataset.csv")

# Display first 5 rows
print(data.head())


# Check dataset size
print("Dataset Shape:", data.shape)

# Check column names
print("Column Names:", data.columns)

# Check dataset information
print(data.info())

# Check missing values
print(data.isnull().sum())

# Fill missing Age values with median
data["Age"] = data["Age"].fillna(data["Age"].median())

# Check missing Age values again
print("Missing Age values:", data["Age"].isnull().sum())


# Fill missing Embarked values with mode
data["Embarked"] = data["Embarked"].fillna(data["Embarked"].mode()[0])

# Check missing values
print("Missing Embarked values:", data["Embarked"].isnull().sum())


# Remove Cabin column
data = data.drop("Cabin", axis=1)

# Check remaining missing values
print(data.isnull().sum())

# Check final dataset shape
print("Final Dataset Shape:", data.shape)

# Check remaining missing values
print("Remaining Missing Values:")
print(data.isnull().sum())

# Display cleaned dataset
print(data.head())

# Separate input features and target
X = data.drop("Survived", axis=1)
y = data["Survived"]

# Display features and target
print("Features Shape:", X.shape)
print("Target Shape:", y.shape)


from sklearn.preprocessing import LabelEncoder

# Convert Sex into numbers
encoder = LabelEncoder()
X["Sex"] = encoder.fit_transform(X["Sex"])

# Display converted values
print(X[["Sex"]].head())

# Remove unnecessary columns
X = X.drop(["PassengerId", "Name", "Ticket"], axis=1)

# Convert categorical columns into numbers
X = pd.get_dummies(X, columns=["Embarked"], dtype=int)

# Check the updated dataset
print(X.head())
print("Updated Features Shape:", X.shape)
# Split data into training and testing
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Display dataset sizes
print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)

# Import Logistic Regression
from sklearn.linear_model import LogisticRegression

# Create the model
model = LogisticRegression(max_iter=1000)

# Train the model
model.fit(X_train, y_train)

print("Model training completed!")

# Import accuracy score
from sklearn.metrics import accuracy_score

# Predict survival using testing data
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Display accuracy
print("Model Accuracy:", accuracy * 100, "%")



# Create sample passenger data
sample_passenger = pd.DataFrame([{
    "Pclass": 3,
    "Sex": 1,
    "Age": 22,
    "SibSp": 1,
    "Parch": 0,
    "Fare": 7.25,
    "Embarked_C": 0,
    "Embarked_Q": 0,
    "Embarked_S": 1
}])

# Predict survival
prediction = model.predict(sample_passenger)

# Display prediction
if prediction[0] == 1:
    print("Passenger Survived!")
else:
    print("Passenger Did Not Survive!")
# Import classification report
from sklearn.metrics import classification_report

# Display model performance
print(classification_report(y_test, y_pred))



import pickle

# Save the trained model
with open("titanic_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model saved successfully!")