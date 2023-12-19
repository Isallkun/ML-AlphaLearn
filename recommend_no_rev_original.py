### ''' recommend_no_rev_original'''
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
import joblib

from google.colab import files
uploaded = files.upload()
df = pd.read_csv('Data2.csv')
df.head()
from ast import literal_eval
df['Choices'] = df['Choices'].apply(literal_eval)

# Combine choices into a single text column
df['CombinedChoices'] = df['Choices'].apply(lambda x: ' '.join(x))
x = df['CombinedChoices']

vectorizer = TfidfVectorizer(stop_words='english')  # Adjust as needed
X = vectorizer.fit_transform(df['CombinedChoices'])
# Train the model to predict cosine similarity as a binary classification problem
similarity_labels = cosine_similarity(X)

# Convert sparse matrix X to a TensorFlow sparse tensor
X_coo = X.tocoo()
X_reordered = tf.sparse.reorder(
    tf.SparseTensor(
        indices=np.vstack((X_coo.row, X_coo.col)).T,
        values=X_coo.data,
        dense_shape=X_coo.shape
    )
)

# Flatten upper triangular part and convert to a 1D array
y_train = similarity_labels[np.triu_indices(similarity_labels.shape[0], k=1)].reshape(-1, 1)

# Ensure that y_train has the same number of rows as X_reordered
y_train = y_train[:X_reordered.shape[0]]

# Convert y_train to a dense tensor
y_train_tensor = tf.convert_to_tensor(y_train, dtype=tf.float32)

# Check the dimensions
print("X_reordered shape:", X_reordered.shape)
print("y_train_tensor shape:", y_train_tensor.shape)
# Update the model architecture to match the dimensions
model = Sequential()
model.add(Dense(8, input_shape=(X_reordered.shape[1],), activation='relu'))
model.add(Dense(1, activation='sigmoid'))  # Output layer

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Print the model summary to see the architecture and parameter count
model.summary()

# Train the model
model.fit(X_reordered, y_train_tensor, epochs=50, batch_size=8)
import random

while True:
    # Randomly choose a question
    random_question_index = random.choice(df.index)
    random_question = df.loc[random_question_index, 'Question']
    choices = df.loc[random_question_index, 'Choices']
    correct_answer = df.loc[random_question_index, 'Correct Answer']

    # Print the random question and choices
    print("Question:", random_question)
    print("Choices:", choices)

    # Assume user provides an answer
    user_answer = input("Enter your answer: ")

    # Check if the user's answer is correct
    if user_answer == correct_answer:
        print("Your answer is correct!")
    else:
        print("Your answer is incorrect.")

        # Use the model to recommend a video link based on the user's incorrect answer
        user_input_vectorized = vectorizer.transform([random_question])
        user_input_reordered = tf.sparse.reorder(tf.SparseTensor(
            indices=np.vstack((user_input_vectorized.tocoo().row, user_input_vectorized.tocoo().col)).T,
            values=user_input_vectorized.tocoo().data,
            dense_shape=user_input_vectorized.tocoo().shape
        ))
        predicted_similarity = model.predict(user_input_reordered)

        # Find the most similar questions
        similar_questions_indices = np.argsort(predicted_similarity[:, 0])[::-1]  # Sort in descending order
        top_k = 1  # Recommend the top 1 question
        top_question_index = similar_questions_indices[0]

        # Get the recommended video link based on the similar question
        recommended_video_link = df.loc[top_question_index, 'VidioLink']  # Note: correct the typo in 'VidioLink'
        print(f"Recommended Video Link for Incorrect Answer: {recommended_video_link}")

    # Ask if the user wants to continue
    continue_playing = input("Do you want to continue? (yes/no): ").lower()
    if continue_playing != 'yes':
        print("Exiting...")
        break

# The code below is to save your model as a .h5 file.
#from tensorflow.keras.models import save_model
# The code below is to save your model as a .h5 file.
if __name__ == '__main__':
    # DO NOT CHANGE THIS CODE
    model.save('recommender_no_rev.h5')

from keras.models import load_model
model = load_model('recommender_no_rev.h5')
print("model loaded")

# Save your model
#joblib.dump(model, 'recommend.pkl')
#print("Model dumped!")

# Load the model that you just saved
#lr = joblib.load('recommend.pkl')

# Saving the data columns from training
model_columns = list(df.columns)
joblib.dump(model_columns, 'recommend_columns.pkl')
#print("Models columns dumped!")
