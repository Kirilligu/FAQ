from sentence_transformers import SentenceTransformer
import numpy as np
import tensorflow as tf
import os

model_name = "DeepPavlov/rubert-base-cased"
sentence_model = SentenceTransformer(model_name)

def load_data():
    with open("data/questions.txt", "r", encoding="utf-8") as f:
        questions = [line.strip() for line in f]
    with open("data/answers.txt", "r", encoding="utf-8") as f:
        answers = [line.strip() for line in f]
    return questions, answers

questions, answers = load_data()
def get_embeddings(texts):
    embeddings = sentence_model.encode(texts)
    return embeddings

question_embeddings = get_embeddings(questions)
answer_embeddings = get_embeddings(answers)
X = np.array([np.concatenate([q_emb, a_emb]) for q_emb in question_embeddings for a_emb in answer_embeddings])
Y = np.array([1 if i == j else 0 for i in range(len(questions)) for j in range(len(answers))])

model_path = "faq_model.h5"
if os.path.exists(model_path):
    keras_model = tf.keras.models.load_model(model_path)
    print("Модель загружена.")
else:
    keras_model = tf.keras.Sequential([
        tf.keras.layers.Dense(300, activation='relu', input_shape=(1536,)),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    keras_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    keras_model.fit(X, Y, epochs=100, batch_size=32)
    keras_model.save(model_path)
    print(f"Модель сохранена в файл: {model_path}")

def chat():
    print("Введите вопрос или 'exit' для выхода:")

    while True:
        user_input = input().strip()
        if user_input.lower() == 'exit':
            print("Завершаем программу...")
            break
        user_emb = get_embeddings([user_input])[0]
        scores = [keras_model.predict(np.concatenate([user_emb, a_emb]).reshape(1, -1), verbose=0)[0][0] for a_emb in
                  answer_embeddings]
        print("Ответ:", answers[np.argmax(scores)])
chat()
