import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import numpy as np
import pickle

# Đọc dataset
data = pd.read_csv("Conversation.csv")
questions = data["question"].tolist()
answers = data["answer"].tolist()

# Tiền xử lý văn bản
def clean_text(text):
    if isinstance(text, str):  # Kiểm tra xem text có phải là chuỗi không
        text = text.lower()
        text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
        return text
    return ""  # Trả về chuỗi rỗng nếu không phải chuỗi

questions = [clean_text(q) for q in questions]

# Vector hóa câu hỏi bằng TF-IDF
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(questions)

# Khởi tạo và fit mô hình KNN
k = 1  # Số láng giềng gần nhất
knn = NearestNeighbors(n_neighbors=k, metric="cosine")
knn.fit(tfidf_matrix)

# Hàm tìm câu trả lời
def get_response(user_input):
    # Tiền xử lý input
    user_input = clean_text(user_input)
    # Vector hóa input
    user_tfidf = vectorizer.transform([user_input])
    # Tìm k láng giềng gần nhất
    distances, indices = knn.kneighbors(user_tfidf)
    # Lấy câu trả lời tương ứng với câu hỏi gần nhất
    return answers[indices[0][0]]

# Lưu các thành phần để dùng trong Django
pickle.dump(vectorizer, open("tfidf_vectorizer.pkl", "wb"))
pickle.dump(knn, open("knn_model.pkl", "wb"))
pickle.dump(tfidf_matrix, open("tfidf_matrix.pkl", "wb"))
pickle.dump(answers, open("answers.pkl", "wb"))

# Test chatbot
print("Nhập câu hỏi (gõ 'quit' để thoát):")
while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        break
    response = get_response(user_input)
    print("Bot:", response)