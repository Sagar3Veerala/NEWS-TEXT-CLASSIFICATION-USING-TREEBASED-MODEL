# News Text Classification Project

This project performs text classification on a news dataset to categorize news articles into different categories (World, Sports, Business, and Sci/Tech) using a deep learning approach based on LSTMs and utilizing various data preprocessing and visualization techniques.

## Table of Contents

1.  [Project Description](#project-description)
2.  [Installation](#installation)
3.  [Usage](#usage)
4.  [Dataset](#dataset)
5.  [Code Structure](#code-structure)
6.  [Dependencies](#dependencies)
7.  [Results](#results)
8.  [Contributing](#contributing)
9.  [License](#license)

## Project Description

This project aims to build a text classification model that can accurately categorize news articles.  The pipeline includes the following steps:

*   **Data Loading and Exploration:** Loading the dataset and performing initial exploratory data analysis (EDA) to understand the data distribution and characteristics.
*   **Text Preprocessing:**  Cleaning and preparing the text data for modeling, including tokenization, lowercasing, stop word removal, and stemming.
*   **Feature Extraction:**  Converting text data into numerical features that can be used by the model. This project uses tokenization and padding.
*   **Model Building:**  Creating an LSTM-based deep learning model for text classification.
*   **Model Training and Evaluation:**  Training the model on the training data and evaluating its performance on the test data using metrics such as accuracy, precision, recall, and F1-score.
*   **Visualization:**  Visualizing the data and model performance using various plots and charts, such as bar plots, box plots, word clouds, and confusion matrices.

## Installation

1.  **Clone the repository:**

    ```bash
    git clone <repository_url>
    cd news-text-classification
    ```

2.  **Create a virtual environment (optional but recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Linux/macOS
    venv\Scripts\activate  # On Windows
    ```

3.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

    **Note:**  Create a `requirements.txt` file with the following content (or similar, based on your environment):

    ```
    numpy
    pandas
    wordcloud
    matplotlib
    seaborn
    plotly
    nltk
    scikit-learn
    tensorflow
    ```

## Usage

1.  **Download the Dataset:** Place the `train.csv` dataset (mentioned below) into a directory named `Dataset` located at the same level as your script.  The directory structure should look like this:

    ```
    news-text-classification/
      - Dataset/
        - train.csv
      - news_classification.py
      - README.md
      - requirements.txt
      - ...
    ```

2.  **Run the script:**

    ```bash
    python news_classification.py  # Or whatever you named your script
    ```

3.  **View the Results:** The script will output various visualizations and performance metrics to the console.

## Dataset

This project uses the AG NEWS dataset, which can be found on Kaggle or other data sources. Ensure you have the `train.csv` file from the AG NEWS dataset.  It contains the text of news articles and their corresponding category labels.

*   **Description:**  The AG News corpus consists of news articles from more than 2000 news sources.
*   **Categories:**
    *   World
    *   Sports
    *   Business
    *   Sci/Tech
*   **Structure:** The dataset contains two main text columns. Title and news article.

## Code Structure

*   `news_classification.py`: The main Python script that contains the code for data loading, preprocessing, model building, training, evaluation, and visualization.  Rename this to whatever you named your script.
*   `Dataset/`:  Directory containing the AG NEWS `train.csv` dataset.  (You will need to create this directory and download the dataset.)
*   `requirements.txt`: File listing the Python packages required to run the script.

## Dependencies

*   Python (3.6 or higher)
*   NumPy
*   Pandas
*   WordCloud
*   Matplotlib
*   Seaborn
*   Plotly
*   NLTK (Natural Language Toolkit)
*   Scikit-learn (sklearn)
*   TensorFlow

These dependencies are listed in the `requirements.txt` file.  You can install them using `pip install -r requirements.txt`.

## Results

The results of the project will be displayed as:

*   **Visualizations:** Bar plots of the class distribution, box plots of description lengths, word clouds for each category, and a confusion matrix showing the model's performance.
*   **Performance Metrics:**  A table summarizing the model's accuracy, precision, recall, and F1-score on the test set.

The specific performance metrics will depend on the random initialization of the model and the dataset used.

.
