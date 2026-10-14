import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the Palmer Penguins dataset
url = ( 
    "https://raw.githubusercontent.com/"
    "allisonhorst/palmerpenguins/main/inst/extdata/penguins.csv"
)

penguins = pd.read_csv(url)

# Define features and target variable
features = [
    "bill_length_mm",
    "bill_depth_mm",
    "flipper_length_mm",
    "body_mass_g",
]

target = "species"

# Drop rows with missing values in the features or target variable
penguins = penguins.dropna(subset=features + [target])

# Separate the features from the target variable
X = penguins[features]
y = penguins[target]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

# Create and train the Decision Tree Classifier
model = DecisionTreeClassifier(max_depth=5 , random_state=42)

model.fit(X_train, y_train)

# Make predictions on the test set
predictions = model.predict(X_test)

# Measure the results
print(f"Test Accuracy: {accuracy_score(y_test, predictions):.2%}")
print("\nClassification Report:") 
print(classification_report(y_test, predictions))
print("Confusion Matrix (rows = actual species, collumns = predicated species):")
print(confusion_matrix(y_test, predictions))
print("Species order:", list(model.classes_))

# Make a prediction for one sample penguin
sample = pd.DataFrame(
    [[45.0, 15.0, 215.0, 5000.0]],
    columns=features, 
)

predicted_species = model.predict(sample)[0]
print("\nSample penguin prediction:", predicted_species)
