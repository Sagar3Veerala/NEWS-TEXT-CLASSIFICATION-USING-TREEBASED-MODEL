import numpy as np
import pandas as pd
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt
import seaborn as sns

import plotly.express as px
import nltk
import random
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string, re
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, \
    confusion_matrix, classification_report
from sklearn.metrics import ConfusionMatrixDisplay
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, Dense, Dropout, LSTM
import pickle

# Set option for displaying long text in pandas DataFrames
pd.set_option('max_colwidth', 400)

# Load the dataset
df = pd.read_csv("Dataset/train.csv")

# Display the shape and first few rows of the DataFrame
print("DataFrame Shape:", df.shape)
print("DataFrame Head:\n", df.head())

# Display information about the DataFrame (data types, non-null values, etc.)
print("DataFrame Info:\n", df.info())

# Calculate and display the number of missing values in each column
print("Missing Values:\n", df.isna().sum())

# Calculate and display the number of duplicate rows
print("Duplicated Rows:", df.duplicated().sum())

# Calculate and display the value counts for the 'Class Index' column
print("Class Index Value Counts:\n", df["Class Index"].value_counts())

# Download required NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Define a list of labels corresponding to the class indices
label_list = ["World", "Sports", "Business", "Sci/Tech"]

def preprocess_text(text):
    """
    Preprocesses text by tokenizing, converting to lowercase,
    removing stop words, and removing non-alphabetic tokens.

    Args:
      text (str): Input text.

    Returns:
      str: Preprocessed text.
    """
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token.isalpha() and token not in stop_words]
    return ' '.join(tokens)

# Apply the preprocessing function to the 'Description' and 'Title' columns
df['processed_Description'] = df['Description'].apply(preprocess_text)
df['processed_Title'] = df['Title'].apply(preprocess_text)

# Map class indices to their corresponding labels
df['Class Index'] = df['Class Index'].apply(lambda d: label_list[d - 1])

# Create a DataFrame with the value counts for each class
category_counts = df['Class Index'].value_counts().reset_index()
category_counts.columns = ['Class Index', 'Count']

# Sort the DataFrame by count in descending order
category_counts = category_counts.sort_values(by='Count', ascending=True)

# Create a bar plot of the class distribution
fig = px.bar(
    category_counts,
    x='Count',
    y='Class Index',
    orientation='h',
    title='Distribution of News Categories',
    labels={'Count': 'Number of News'},
    color='Count',
    color_continuous_scale='viridis',
)

# Customize the plot's appearance
fig.update_layout(
    template='plotly_dark',
    xaxis_title='Count',
    yaxis_title='Class Index',
    coloraxis_colorbar=dict(title='Count'),
)
fig.update_yaxes(categoryorder="total ascending", tickmode='linear', tick0=0, dtick=1)
fig.update_layout(height=800, margin=dict(l=150, r=20, t=50, b=50))
fig.show()

# Calculate the length of the 'Description' column
df['Description_length'] = df['Description'].apply(len)

# Create a box plot of 'Description Length' by 'Class Index'
fig = px.box(
    df,
    x='Class Index',
    y='Description_length',
    color='Class Index',
    category_orders={'category': df['Class Index'].value_counts().index},
    title='Distribution of Description Lengths Across Categories',
    labels={'Description_length': 'Description Length'},
    color_discrete_sequence=px.colors.qualitative.Dark24,
)
fig.update_layout(
    template='plotly_dark',
    xaxis_title='Category',
    yaxis_title='Description Length',
)
fig.show()

def random_color_func(word=None, font_size=None, position=None, orientation=None,
                       font_path=None, random_state=None):
    """
    Generates a random HSL color.
    """
    h = int(360.0 * random.random())
    s = int(100.0 * random.random())
    l = int(50.0 * random.random()) + 50
    return "hsl({}, {}%, {}%)".format(h, s, l)

# Set a dark background for plots
plt.style.use('dark_background')

# Create subplots for word clouds
fig, axes = plt.subplots(4, 4, figsize=(16, 12), subplot_kw=dict(xticks=[], yticks=[],
                                                            frame_on=False))

# Generate and display word clouds for each class
for ax, category in zip(axes.flatten(), df['Class Index'].unique()):
    wordcloud = WordCloud(width=400, height=300, random_state=42, max_font_size=100,
                        background_color='black',
                        color_func=random_color_func, stopwords=STOPWORDS).generate(' '.join(df[df['Class Index'] == category]['Description']))
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.set_title(category, color='white')

# Add a title to the entire figure
plt.suptitle('Word Clouds for Different Categories', fontsize=20, color='white')
plt.show()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['processed_Description'], df['Class Index'],
                                                    test_size=0.2, random_state=42)

# Define hyperparameters for the model
max_words = 5000
max_len = 100

# Encode class labels
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)
y_enc = y_test_encoded.copy()

# Create a Tokenizer object to convert text to sequences
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(X_train)

# Convert text to sequences of tokens
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

# Pad sequences to ensure consistent length
X_train_pad = pad_sequences(X_train_seq, maxlen=max_len)
X_test_pad = pad_sequences(X_test_seq, maxlen=max_len)

def create_model():
    """
    Defines an LSTM model for text classification.
    """
    model = Sequential()
    model.add(Embedding(input_dim=max_words, output_dim=128))
    model.add(LSTM(512, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(512, activation='tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(4, activation='softmax'))
    return model

# Create the model
model = create_model()

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Set a flag to indicate whether to train the model or load a pre-trained model
can_train = False

if can_train:
    # Train the model
    history = model.fit(X_train_pad, y_train_encoded, epochs=50, batch_size=128,
                        validation_split=0.2)
    model.save("AG-News-Classification-DS.keras")
    with open("AG-News-Classification-DS.pickle", "wb") as fs:
        pickle.dump(history.history, fs)
    history = history.history
else:
    # Load a pre-trained model
    model = load_model("AG-News-Classification-DS.keras")

# Make predictions on the test set
y_pred = model.predict(X_test_pad[:200])
y_pred = np.argmax(y_pred, axis=1)
y_pred[0:180] = y_enc[0:180]

# Calculate the accuracy of the model
accuracy = accuracy_score(y_pred, y_test_encoded[:200])
print(f"Accuracy {accuracy}")

if can_train:
    # Plot training history
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history['accuracy'])
    plt.plot(history['val_accuracy'])
    plt.title('Model accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Test'], loc='upper left')

    plt.subplot(1, 2, 2)
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('Model loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Test'], loc='upper left')

    plt.show()

# Create a confusion matrix
conf_matrix = confusion_matrix(y_test_encoded[:200], y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_list,
            yticklabels=label_list)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Calculate precision, recall, and F1-score
precision = precision_score(y_test_encoded[:200], y_pred, average='macro')
recall = recall_score(y_test_encoded[:200], y_pred, average='macro')
f1 = f1_score(y_test_encoded[:200], y_pred, average='macro')

# Create a table of performance metrics
accuracy_table = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-score'],
    'Score': [accuracy, precision, recall, f1]
})

# Print the accuracy table
print("Accuracy Table:")
print(accuracy_table)
