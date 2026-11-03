<div align="center">

# Book Recommender ML Model

**A book recommender built as three notebooks: data preparation and EDA, a Keras embedding network that predicts ratings, and a top-5 recommendation step.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python](https://img.shields.io/badge/Python-3-3776AB?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-FF6F00?logo=tensorflow&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-150458?logo=pandas&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)

</div>

This is a practice project on recommender systems. It explores a book ratings dataset (books, users and ratings tables), prepares train and test splits, trains a neural collaborative-filtering model with user and book embeddings, and uses it to list the five highest predicted books a given user has not rated yet. The model is a first iteration and the notebooks are candid about its limits.

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [How It Works](#how-it-works)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Methodology and Results](#methodology-and-results)
- [Limitations and Next Steps](#limitations-and-next-steps)
- [License](#license)
- [Author](#author)

## Overview

The task is to predict the rating a user would give to a book they have not rated, and to recommend the books with the highest predicted rating. The work is split into three notebooks that are meant to be run in order:

1. `001-book-recommender-test-preparation.ipynb` loads, cleans and explores the data and writes the train and test files.
2. `002-book-recommender-test-training.ipynb` builds, trains, evaluates and saves the Keras model.
3. `003-book-recommender-test-prediction.ipynb` loads the saved model and produces five recommendations for one user.

## Key Features

- Cleaning of the books, users and ratings tables: numeric conversion of publication year and age, removal of image URL columns, text normalisation and null handling.
- Exploratory analysis: publisher and author frequencies, publication year outliers, title length against rating, rating distribution, ratings per user and per book, age distribution and age groups, users by country.
- ISBN validity check with `isbnlib`.
- 75 / 25 split of the ratings table (`random_state=42`), joined back to the users and books tables and saved as CSV files.
- Neural collaborative filtering in Keras: user and book embeddings of size 15, concatenated and fed to a dense layer of 128 units with a linear output, trained with mean squared error.
- Prediction notebook that scores all unrated books for a user and returns the top five.

## How It Works

```mermaid
flowchart LR
    A[Books Users Ratings CSV] --> B[001 Cleaning and EDA]
    B --> C[Train and test CSV files]
    C --> D[002 Embedding network in Keras]
    D --> E[Saved .h5 model]
    E --> F[003 Top 5 recommendations]
```

Model architecture used in notebook 002:

| Layer | Detail |
|-------|--------|
| Inputs | One integer user index and one integer book index, produced with `LabelEncoder` |
| Embeddings | Size 15 for users and for books |
| Merge | Flatten both vectors and concatenate |
| Hidden | Dense, 128 units, ReLU |
| Output | Dense, 1 unit, linear (predicted rating) |
| Training | Adam optimiser, mean squared error loss, 5 epochs, batch size 32 |

## Tech Stack

| Area | Technology |
|------|------------|
| Language | Python 3 |
| Data handling | pandas, NumPy |
| Visualisation | Matplotlib, seaborn |
| Data quality | isbnlib |
| Preprocessing and splitting | scikit-learn (`LabelEncoder`, `train_test_split`) |
| Deep learning | TensorFlow / Keras |
| Environment | Jupyter Notebook |

## Project Structure

```text
.
├── 001-book-recommender-test-preparation.ipynb   # Loading, cleaning, EDA, train/test split
├── 002-book-recommender-test-training.ipynb      # Keras embedding model: training and evaluation
├── 003-book-recommender-test-prediction.ipynb    # Loads the model and recommends 5 books for a user
├── LICENSE
└── README.md
```

Files created when the notebooks are run, and not stored in the repository: `train_usuarios_calificaciones.csv`, `test_usuarios_calificaciones.csv`, `train_libros_calificaciones.csv`, `test_libros_calificaciones.csv` and `training_model_try_recommender.h5`.

## Dataset

The notebooks expect three CSV files in the working directory, which are not included in the repository:

| File | Columns seen in the notebook output | Records |
|------|-------------------------------------|---------|
| `Books.csv` | ISBN, Book-Title, Book-Author, Year-Of-Publication, Publisher, Image-URL-S, Image-URL-M, Image-URL-L | 271,360 |
| `Ratings.csv` | User-ID, ISBN, Book-Rating | not printed in the saved output |
| `Users.csv` | User-ID, Location, Age | not printed in the saved output |

The repository does not state where the files come from. The table layout matches the public Book-Crossing ratings dataset that is commonly distributed on Kaggle, but this is an inference: obtain the three files from a source you trust and place them in the repository root.

## Getting Started

### Prerequisites

- Python 3 with `pip`
- The three data files described above
- Patience for training: the saved output of notebook 002 shows about 30 minutes per epoch on the machine that ran it

### Installation

```bash
git clone https://github.com/LargoLujan/book-recommender-ml-model.git
cd book-recommender-ml-model
python -m venv .venv
source .venv/bin/activate        # On Windows: .venv\Scripts\activate
pip install pandas numpy matplotlib seaborn scikit-learn isbnlib tensorflow jupyter
```

The first notebook also imports `Reader`, `Dataset`, `SVD` and `cross_validate` from `surprise`, so install `scikit-surprise` as well (`pip install scikit-surprise`) or remove that import line. These objects are not used later in the notebook.

### Running

Place `Books.csv`, `Ratings.csv` and `Users.csv` in the repository root, then run the notebooks in order:

```bash
jupyter notebook
```

Open and run `001-...`, then `002-...`, then `003-...`. Each notebook reads files produced by the previous one.

## Usage

In notebook 003 the user is chosen with the `user_id` variable (the saved run uses `10`). The notebook encodes every book the user has not rated, predicts a rating for each, sorts the predictions and prints the ISBN and title of the five highest. Change `user_id` to get recommendations for another user. The `user_id` must be one of the encoded indices used for training.

## Methodology and Results

All figures come from the saved outputs of the notebooks.

**Notebook 001 (data preparation)**

- The books table has 271,360 records. Publication years contain invalid values: the saved statistics show a minimum of 0 and a maximum of 2050, and the notebook lists the books with years after 2023.
- Most prolific authors in the books table: Agatha Christie (632 books), William Shakespeare (567) and Stephen King (524).
- Ratings per user (105,283 users in the printed summary): median of 1, mean of about 10.9, maximum of 13,602.
- Most rated book by count of ratings: ISBN 0971880107 with 2,502 ratings.
- Users by country: the USA leads (139,183 users), followed by Canada (21,556) and the United Kingdom (18,286).

**Notebook 002 (model)**

- Training loss (MSE) by epoch: 11.72, 9.38, 7.90, 6.97 and 6.19.
- Loss (MSE) on the test set after the fifth epoch: 12.96.
- The gap between training and test loss indicates that the model overfits, which fits the very sparse ratings per user seen in the exploration. The notebook plots a histogram of the prediction errors (test predictions minus true ratings) and interprets it qualitatively.

**Notebook 003 (recommendation)**

- The saved output lists five recommended titles for the chosen user (for example a parenting guide, a book on HTML and a religious classic). The notebook text itself notes that results are not varied, so this step is a demonstration of the mechanics rather than a validated recommender. The print statement in that cell still says "usuario 12345" although `user_id` is 10.

## Limitations and Next Steps

- The model is a first baseline; no hyper-parameter tuning, regularisation or comparison with other approaches is included.
- Ranking quality (for example precision at k) is not measured; only MSE is reported.
- Ratings of 0 are not treated separately in the code shown.
- The data has many invalid ISBNs and publication years that are only detected, not repaired, in notebook 001.
- Possible next steps, as suggested in the original project notes: matrix-factorisation or content-based models, a more robust ISBN cleaning step and additional evaluation metrics.

## License

Distributed under the MIT License. See [LICENSE](LICENSE) for details.

## Author

**Manuel Luján Vílchez** — Senior Data Engineer

[![GitHub](https://img.shields.io/badge/GitHub-LargoLujan-181717?logo=github)](https://github.com/LargoLujan)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Manuel%20Luj%C3%A1n%20V%C3%ADlchez-0A66C2?logo=linkedin)](https://www.linkedin.com/in/manuel-lujan-vilchez)
