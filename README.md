# Penguin Species Classifier

A beginner machine learning project that uses penguin measurements to predict a penguin’s species. It trains a decision tree with the Palmer Penguins dataset and evaluates its predictions on data the model did not train on.

## What the program does

- Loads penguin data from a public CSV file.
- Uses bill length, bill depth, flipper length, and body mass as input features.
- Removes rows that are missing required measurements or species labels.
- Splits the data into training and testing sets.
- Trains a decision tree classifier.
- Reports test accuracy, a classification report, and a confusion matrix.
- Predicts the species for one sample penguin.

## Requirements

- Python 3
- pandas
- scikit-learn
- An internet connection for downloading the dataset when the program runs

## Setup

Open a terminal in the project folder and install the required packages:

```bash
python3 -m pip install pandas scikit-learn
```

On Windows, if `python3` is not recognized, use:

```bash
py -m pip install pandas scikit-learn
```

## Run the project

Run the program from the project folder:

```bash
python3 penguin_classifier.py
```

On Windows, you can use:

```bash
py penguin_classifier.py
```

The program prints evaluation results and a sample species prediction in the terminal.

## How it works

The program uses four measurements—bill length, bill depth, flipper length, and body mass—to predict the species label. It reserves some examples for testing, trains a decision tree on the remaining examples, and compares its predictions with the known labels in the test set.

The sample penguin measurements near the end of the program are examples. Change them to try another prediction, keeping the same feature order and units.

## Limitations

This is a learning project, and its predictions can be wrong. Its results reflect the dataset and test split used here; they do not guarantee accurate predictions for every penguin or for data collected in different conditions.

## Data

The program loads the Palmer Penguins dataset from the [Palmer Penguins GitHub repository](https://github.com/allisonhorst/palmerpenguins).