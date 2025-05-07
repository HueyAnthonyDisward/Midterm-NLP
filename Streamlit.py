import random
import numpy as np
import streamlit as st
import nlpaug.augmenter.word as naw
import requests
from bs4 import BeautifulSoup
import nltk
from sklearn.tree import DecisionTreeClassifier
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelBinarizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import Word2Vec, FastText
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import torch
from io import BytesIO
from sklearn.svm import SVC
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel
import base64

# Tải dữ liệu NLTK
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# CSS tùy chỉnh để làm đẹp giao diện
st.markdown("""
    <style>
    /* Tùy chỉnh nền và font chữ */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        font-family: 'Segoe UI', sans-serif;
    }

    /* Tiêu đề chính */
    h1 {
        color: #2c3e50;
        font-weight: 700;
        text-align: center;
        margin-bottom: 20px;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.1);
    }

    /* Tiêu đề phụ */
    h2, h3 {
        color: #34495e;
        font-weight: 600;
        margin-top: 20px;
        border-left: 4px solid #3498db;
        padding-left: 10px;
    }

    /* Nút */
    .stButton>button {
        background-color: #3498db;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #2980b9;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }

    /* Text input và selectbox */
    .stTextInput, .stSelectbox {
        border-radius: 8px;
        background-color: #ffffff;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }

    /* Tùy chỉnh container */
    .stContainer {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }

    /* Thanh sidebar */
    .css-1d391kg {
        background-color: #2c3e50;
        color: white;
    }
    .css-1d391kg h2 {
        color: white;
        border-left: none;
    }

    /* Căn giữa văn bản */
    .center-text {
        text-align: center;
    }

    /* Thông báo */
    .stAlert {
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    </style>
""", unsafe_allow_html=True)

# Tiêu đề chính
st.markdown("<h1>Ứng dụng NLP - Nguyễn Trung Hiếu</h1>", unsafe_allow_html=True)

# Thanh sidebar để điều hướng
st.sidebar.markdown("<h2>Điều hướng</h2>", unsafe_allow_html=True)
section = st.sidebar.radio("Chọn chức năng:", [
    "Cào dữ liệu",
    "Tăng cường dữ liệu",
    "Tiền xử lý văn bản",
    "Mô hình hóa văn bản",
    "Phân loại văn bản",
    "Gợi ý sản phẩm",
    "Trò chuyện với AI"
])

# Khởi tạo session state
if 'text_data' not in st.session_state:
    st.session_state.text_data = ""
if 'model_results' not in st.session_state:
    st.session_state.model_results = {}
if 'current_model' not in st.session_state:
    st.session_state.current_model = None
if 'current_vectorizer' not in st.session_state:
    st.session_state.current_vectorizer = None
if 'current_vectorization_method' not in st.session_state:
    st.session_state.current_vectorization_method = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []


# Hàm hiển thị thông báo thành công
def show_success(message):
    st.success(message)
    st.balloons()


# Hàm hiển thị ma trận nhầm lẫn
def display_confusion_matrix(matrix, title):
    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', cbar=True)
    plt.title(title)
    plt.ylabel('Nhãn thực tế')
    plt.xlabel('Nhãn dự đoán')
    buffer = BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight')
    plt.close()
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


# Phần 1: Cào dữ liệu
if section == "Cào dữ liệu":
    with st.container():
        st.markdown("<h2>1️⃣ Cào dữ liệu</h2>", unsafe_allow_html=True)
        option = st.radio("Chọn nguồn dữ liệu:", ["Cào từ web", "Nhập văn bản", "Tải file TXT"])

        if option == "Cào từ web":
            url = st.text_input("Nhập URL:", "https://vnexpress.net/", placeholder="Nhập URL hợp lệ...")
            if st.button("Cào dữ liệu"):
                with st.spinner("Đang cào dữ liệu..."):
                    response = requests.get(url)
                    if response.status_code == 200:
                        soup = BeautifulSoup(response.text, 'html.parser')
                        paragraphs = soup.find_all('p')
                        st.session_state.text_data = " ".join([p.get_text() for p in paragraphs])
                        show_success("Cào dữ liệu thành công!")
                        st.write("📄 **Dữ liệu cào được (500 ký tự đầu tiên):**")
                        st.markdown(f"<div class='stContainer'>{st.session_state.text_data[:500]}</div>",
                                    unsafe_allow_html=True)
                    else:
                        st.error("Không thể cào dữ liệu từ URL!")

        elif option == "Nhập văn bản":
            st.session_state.text_data = st.text_area("Nhập văn bản:", st.session_state.text_data,
                                                      placeholder="Nhập văn bản của bạn...")
            if st.session_state.text_data:
                show_success("Văn bản đã được lưu!")

        elif option == "Tải file TXT":
            uploaded_file = st.file_uploader("Chọn file TXT", type="txt")
            if uploaded_file:
                st.session_state.text_data = uploaded_file.read().decode("utf-8")
                show_success("Tải file thành công!")
                st.write("📄 **Nội dung file (500 ký tự đầu tiên):**")
                st.markdown(f"<div class='stContainer'>{st.session_state.text_data[:500]}</div>",
                            unsafe_allow_html=True)

# Phần 2: Tăng cường dữ liệu
elif section == "Tăng cường dữ liệu":
    with st.container():
        st.markdown("<h2>2️⃣ Tăng cường dữ liệu</h2>", unsafe_allow_html=True)

        # Khởi tạo các công cụ tăng cường
        synonym_aug = naw.SynonymAug(aug_src='wordnet', aug_p=0.2)
        random_insert_aug = naw.RandomWordAug(action="insert", aug_p=0.2)
        random_swap_aug = naw.RandomWordAug(action="swap", aug_p=0.2)
        random_delete_aug = naw.RandomWordAug(action="delete", aug_p=0.2)

        augment_method = st.radio("Chọn phương pháp tăng cường:", [
            "Thay thế từ đồng nghĩa",
            "Chèn từ ngẫu nhiên",
            "Đổi chỗ từ",
            "Xóa từ ngẫu nhiên"
        ])

        if st.button("Tăng cường dữ liệu") and st.session_state.text_data:
            augmented_text = st.session_state.text_data
            if augment_method == "Thay thế từ đồng nghĩa":
                augmented_text = synonym_aug.augment(augmented_text)
            elif augment_method == "Chèn từ ngẫu nhiên":
                augmented_text = random_insert_aug.augment(augmented_text)
            elif augment_method == "Đổi chỗ từ":
                augmented_text = random_swap_aug.augment(augmented_text)
            elif augment_method == "Xóa từ ngẫu nhiên":
                augmented_text = random_delete_aug.augment(augmented_text)
            st.session_state.augmented_text = augmented_text
            show_success("Tăng cường dữ liệu thành công!")
            st.write("🔹 **Kết quả:**")
            st.markdown(f"<div class='stContainer'>{augmented_text}</div>", unsafe_allow_html=True)

            if st.button("Lưu kết quả"):
                st.session_state.text_data = st.session_state.augmented_text
                show_success("Kết quả đã được lưu vào dữ liệu chính!")

# Phần 3: Tiền xử lý văn bản
elif section == "Tiền xử lý văn bản":
    with st.container():
        st.markdown("<h2>3️⃣ Tiền xử lý văn bản</h2>", unsafe_allow_html=True)
        option_process = st.selectbox("Chọn phương pháp:", [
            "Cleaning",
            "Tokenization",
            "Stopwords Removal",
            "Lemmatization"
        ])

        if st.button("Xử lý") and st.session_state.text_data:
            processed_text = ""
            if option_process == "Cleaning":
                processed_text = st.session_state.text_data.replace(".", "").replace(",", "")
            elif option_process == "Tokenization":
                processed_text = word_tokenize(st.session_state.text_data)
            elif option_process == "Stopwords Removal":
                stop_words = set(stopwords.words("english"))
                words = word_tokenize(st.session_state.text_data)
                processed_text = [word for word in words if word.lower() not in stop_words]
            elif option_process == "Lemmatization":
                lemmatizer = WordNetLemmatizer()
                words = word_tokenize(st.session_state.text_data)
                processed_text = [lemmatizer.lemmatize(word) for word in words]

            st.session_state.processed_text = processed_text
            show_success("Xử lý văn bản thành công!")
            st.write("🔹 **Kết quả:**")
            st.markdown(f"<div class='stContainer'>{processed_text}</div>", unsafe_allow_html=True)

            if st.button("Lưu kết quả"):
                st.session_state.text_data = " ".join(processed_text) if isinstance(processed_text,
                                                                                    list) else processed_text
                show_success("Kết quả đã được lưu!")

# Phần 4: Mô hình hóa văn bản
elif section == "Mô hình hóa văn bản":
    with st.container():
        st.markdown("<h2>4️⃣ Mô hình hóa văn bản</h2>", unsafe_allow_html=True)
        if st.session_state.text_data:
            words = st.session_state.text_data.split()

            col1, col2 = st.columns(2)
            with col1:
                i = st.slider("Chọn từ bắt đầu (i):", 1, len(words), 1)
            with col2:
                j = st.slider("Chọn từ kết thúc (j):", i, len(words), len(words))

            selected_text = " ".join(words[i - 1:j])

            method = st.selectbox("Chọn phương pháp:", [
                "One-Hot Encoding", "Bag of Words", "TF-IDF", "Bag of N-Gram",
                "Word2Vec", "CBOW", "BERT", "FastText"
            ])

            if st.button("Chuyển đổi"):
                if method == "One-Hot Encoding":
                    encoder = LabelBinarizer()
                    selected_words = selected_text.split()
                    encoded = encoder.fit_transform(selected_words)
                    st.write("🔹 **Kết quả One-Hot Encoding:**")
                    for word, encoding in zip(selected_words, encoded):
                        st.markdown(f"<div class='stContainer'>{word}: {list(encoding)}</div>", unsafe_allow_html=True)

                elif method == "Bag of Words":
                    vectorizer = CountVectorizer()
                    bow_matrix = vectorizer.fit_transform([selected_text])
                    words = vectorizer.get_feature_names_out()
                    counts = bow_matrix.toarray()[0]
                    st.write("🔹 **Kết quả Bag of Words:**")
                    for word, count in zip(words, counts):
                        st.markdown(f"<div class='stContainer'>{word}: {count}</div>", unsafe_allow_html=True)

                elif method == "TF-IDF":
                    vectorizer = TfidfVectorizer()
                    tfidf_matrix = vectorizer.fit_transform([selected_text])
                    words = vectorizer.get_feature_names_out()
                    scores = tfidf_matrix.toarray()[0]
                    st.write("🔹 **Kết quả TF-IDF:**")
                    for word, score in zip(words, scores):
                        st.markdown(f"<div class='stContainer'>{word}: {score:.4f}</div>", unsafe_allow_html=True)

                elif method == "Bag of N-Gram":
                    n = 3
                    vectorizer = CountVectorizer(ngram_range=(n, n))
                    ngram_matrix = vectorizer.fit_transform([selected_text])
                    ngrams = vectorizer.get_feature_names_out()
                    counts = ngram_matrix.toarray()[0]
                    st.write("🔹 **Kết quả Bag of N-Gram:**")
                    for ngram, count in zip(ngrams, counts):
                        st.markdown(f"<div class='stContainer'>{ngram}: {count}</div>", unsafe_allow_html=True)

                elif method in ["Word2Vec", "CBOW"]:
                    words_tokenized = [word_tokenize(selected_text.lower())]
                    sg_value = 1 if method == "Word2Vec" else 0
                    model = Word2Vec(words_tokenized, vector_size=100, window=5, min_count=1, workers=4, sg=sg_value)
                    st.write("🔹 **Vector của các từ:**")
                    for word in words_tokenized[0]:
                        st.markdown(f"<div class='stContainer'>{word}: {model.wv[word].tolist()[:5]}...</div>",
                                    unsafe_allow_html=True)

                elif method == "BERT":
                    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
                    model = BertModel.from_pretrained("bert-base-uncased")
                    tokens = tokenizer(selected_text, return_tensors="pt", padding=True, truncation=True)
                    with torch.no_grad():
                        outputs = model(**tokens)
                    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().tolist()
                    st.write("🔹 **Vector BERT (5 giá trị đầu tiên):**")
                    st.markdown(f"<div class='stContainer'>{embedding[:5]}</div>", unsafe_allow_html=True)

                elif method == "FastText":
                    words_tokenized = [word_tokenize(selected_text.lower())]
                    model = FastText(words_tokenized, vector_size=100, window=5, min_count=1, workers=4)
                    st.write("🔹 **Vector của các từ:**")
                    for word in words_tokenized[0]:
                        st.markdown(f"<div class='stContainer'>{word}: {model.wv[word].tolist()[:5]}...</div>",
                                    unsafe_allow_html=True)

                show_success("Chuyển đổi thành công!")
        else:
            st.warning("🔹 Vui lòng nhập đoạn văn bản để thực hiện xử lý.")

# Phần 5: Phân loại văn bản
elif section == "Phân loại văn bản":
    with st.container():
        st.markdown("<h2>5️⃣ Phân loại văn bản</h2>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            vectorization_method = st.selectbox("Chọn phương pháp vector hóa:", [
                "One-Hot Encoding", "Bag of Words", "TF-IDF", "Bag of N-Gram", "Word2Vec", "CBOW"
            ])
        with col2:
            model_type = st.selectbox("Chọn mô hình phân loại:", [
                "Naive Bayes", "SVM", "Logistic Regression", "KNN", "Decision Tree"
            ])

        uploaded_file = st.file_uploader("Tải lên tệp CSV", type="csv")

        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            if 'sentiment' in df.columns and df.shape[1] >= 2:
                text_data = df.iloc[:, 0].astype(str)
                labels = df['sentiment']

                X = None
                vectorizer = None

                if vectorization_method == "One-Hot Encoding":
                    vectorizer = CountVectorizer(binary=True)
                    X = vectorizer.fit_transform(text_data)

                elif vectorization_method == "Bag of Words":
                    vectorizer = CountVectorizer(max_features=5000)
                    X = vectorizer.fit_transform(text_data)

                elif vectorization_method == "TF-IDF":
                    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
                    X = vectorizer.fit_transform(text_data)

                elif vectorization_method == "Bag of N-Gram":
                    n = 3
                    vectorizer = CountVectorizer(ngram_range=(n, n), max_features=5000)
                    X = vectorizer.fit_transform(text_data)

                elif vectorization_method in ["Word2Vec", "CBOW"]:
                    tokenized_texts = [word_tokenize(text.lower()) for text in text_data]
                    sg_value = 1 if vectorization_method == "Word2Vec" else 0
                    model = Word2Vec(tokenized_texts, vector_size=100, window=5, min_count=1, workers=4, sg=sg_value)
                    X = np.array([
                        np.mean([model.wv[word] for word in text if word in model.wv], axis=0)
                        if len([word for word in text if word in model.wv]) > 0 else np.zeros(100)
                        for text in tokenized_texts
                    ])
                    vectorizer = model

                if X is not None and X.shape[0] > 0:
                    if isinstance(X, np.ndarray) and X.ndim == 1:
                        X = X.reshape(-1, 1)

                    if isinstance(X, np.ndarray) and X.shape[1] > 300:
                        svd = TruncatedSVD(n_components=300)
                        X = svd.fit_transform(X)

                    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

                    if st.button("Huấn luyện"):
                        model = None
                        if model_type == "Naive Bayes":
                            if vectorization_method in ["Word2Vec", "CBOW"]:
                                st.error("❌ MultinomialNB không hỗ trợ Word2Vec hoặc CBOW!")
                            else:
                                model = MultinomialNB()
                        elif model_type == "SVM":
                            model = SVC(probability=True)
                        elif model_type == "KNN":
                            k = st.slider("Chọn số láng giềng (K):", 1, 20, 5, step=2)
                            model = KNeighborsClassifier(n_neighbors=k)
                        elif model_type == "Decision Tree":
                            dep = st.slider("Chọn độ sâu tối đa (max_depth):", 1, 50, 10)
                            model = DecisionTreeClassifier(max_depth=dep)
                        else:
                            model = LogisticRegression()

                        if model:
                            model.fit(X_train, y_train)
                            y_pred = model.predict(X_test)
                            accuracy = accuracy_score(y_test, y_pred)
                            report = classification_report(y_test, y_pred)
                            matrix = confusion_matrix(y_test, y_pred)
                            matrix_image = display_confusion_matrix(matrix,
                                                                    f"Ma trận nhầm lẫn ({model_type}, {vectorization_method})")

                            model_key = f"{model_type}_{vectorization_method}_{len(st.session_state.model_results)}"
                            st.session_state.model_results[model_key] = {
                                "accuracy": accuracy,
                                "matrix_image": matrix_image,
                                "report": report,
                                "vectorization_method": vectorization_method,
                                "model_type": model_type
                            }
                            st.session_state.current_model = model
                            st.session_state.current_vectorizer = vectorizer
                            st.session_state.current_vectorization_method = vectorization_method

                            st.write(f"🎯 **Độ chính xác:** {accuracy:.4f}")
                            st.write("📊 **Báo cáo phân loại:**")
                            st.code(report, language="plaintext")
                            st.image(f"data:image/png;base64,{matrix_image}")
                            show_success("Mô hình đã được huấn luyện thành công!")

                else:
                    st.error("❌ Dữ liệu không hợp lệ!")

        if st.session_state.model_results:
            st.markdown("<h3>📊 So sánh các mô hình</h3>", unsafe_allow_html=True)
            accuracies = {key: result["accuracy"] for key, result in st.session_state.model_results.items()}
            model_names = list(accuracies.keys())

            plt.figure(figsize=(10, 6))
            plt.bar(model_names, accuracies.values(), color='skyblue')
            plt.title("So sánh độ chính xác của các mô hình")
            plt.xlabel("Mô hình (Phương pháp vector hóa)")
            plt.ylabel("Độ chính xác")
            plt.xticks(rotation=45, ha='right')
            buffer = BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight')
            plt.close()
            accuracy_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
            st.image(f"data:image/png;base64,{accuracy_image}")

            st.write("🚶 **Ma trận nhầm lẫn của các mô hình:**")
            cols = st.columns(min(len(st.session_state.model_results), 3))
            for idx, (key, result) in enumerate(st.session_state.model_results.items()):
                with cols[idx % 3]:
                    st.write(f"**{key}**")
                    st.image(f"data:image/png;base64,{result['matrix_image']}")
                    st.write(f"Độ chính xác: {result['accuracy']:.4f}")

        st.markdown("<h3>📝 Kiểm tra mô hình</h3>", unsafe_allow_html=True)
        user_input = st.text_area("Nhập văn bản để dự đoán:", placeholder="Nhập văn bản...")
        if user_input and st.session_state.current_model:
            input_vector = None
            if st.session_state.current_vectorization_method == "One-Hot Encoding":
                input_vector = st.session_state.current_vectorizer.transform([user_input])
            elif st.session_state.current_vectorization_method == "Bag of Words":
                input_vector = st.session_state.current_vectorizer.transform([user_input])
            elif st.session_state.current_vectorization_method == "TF-IDF":
                input_vector = st.session_state.current_vectorizer.transform([user_input])
            elif st.session_state.current_vectorization_method == "Bag of N-Gram":
                input_vector = st.session_state.current_vectorizer.transform([user_input])
            elif st.session_state.current_vectorization_method in ["Word2Vec", "CBOW"]:
                tokenized_input = word_tokenize(user_input.lower())
                input_vector = np.mean([st.session_state.current_vectorizer.wv[word] for word in tokenized_input if
                                        word in st.session_state.current_vectorizer.wv], axis=0)
                if input_vector.shape == (100,):
                    input_vector = input_vector.reshape(1, -1)
                else:
                    input_vector = np.zeros((1, 100))

            if input_vector is not None:
                prediction = st.session_state.current_model.predict(input_vector)
                st.write(f"🔮 **Dự đoán cảm xúc:** `{prediction[0]}`")
                show_success("Dự đoán thành công!")
            else:
                st.error("❌ Không thể chuyển đổi văn bản!")

# Phần 6: Gợi ý sản phẩm
elif section == "Gợi ý sản phẩm":
    with st.container():
        st.markdown("<h2>6️⃣ Gợi ý sản phẩm</h2>", unsafe_allow_html=True)
        rating_file = st.file_uploader("Tải lên tệp CSV về sản phẩm", type="csv")

        if rating_file:
            data = pd.read_csv(rating_file)
            st.write("**Dữ liệu ban đầu:**")
            st.dataframe(data.head())

            ratings = data[['name', 'asin', 'rating']].copy()
            ratings['rating'] = pd.to_numeric(ratings['rating'], errors='coerce')
            ratings.dropna(subset=['rating'], inplace=True)
            ratings = ratings.groupby(['name', 'asin'], as_index=False).agg({'rating': 'mean'})
            user_counts = ratings['name'].value_counts()
            valid_users = user_counts[user_counts >= 3].index
            ratings = ratings[ratings['name'].isin(valid_users)]
            rating_matrix = ratings.pivot(index='name', columns='asin', values='rating').fillna(0)

            if rating_matrix.shape[1] >= 5:
                new_user_name = 'Huey'
                all_products = rating_matrix.columns.tolist()
                new_user_products = random.sample(all_products, 5)
                new_user_ratings = [random.randint(1, 5) for _ in range(5)]
                new_user_vector = pd.DataFrame([new_user_ratings], columns=new_user_products, index=[new_user_name])
                new_user_vector = new_user_vector.reindex(columns=rating_matrix.columns, fill_value=0)

                k = min(10, rating_matrix.shape[1])
                svd = TruncatedSVD(n_components=k)
                U_r = svd.fit_transform(rating_matrix)
                Sigma_r = np.diag(svd.singular_values_)
                V_r_T = svd.components_
                r = new_user_vector.values.reshape(1, -1)
                Sigma_r_inv = np.linalg.pinv(Sigma_r)
                U_new = np.dot(np.dot(r, V_r_T.T), Sigma_r_inv)
                similarities = cosine_similarity(U_new, U_r)[0]
                similar_users = np.argsort(similarities)[::-1][:10]
                similar_user_ids = rating_matrix.iloc[similar_users].index.tolist()

                st.write("**10 người dùng gần nhất với Huey:**")
                for i, user in enumerate(similar_user_ids):
                    st.markdown(
                        f"<div class='stContainer'>{i + 1}. {user} - Cosine Similarity: {similarities[similar_users[i]]:.4f}</div>",
                        unsafe_allow_html=True)

                test_user_rated_products = set(new_user_products)
                recommended_products = []
                seen_products = set()

                for user in similar_user_ids:
                    user_top_products = ratings[(ratings['name'] == user) & (ratings['rating'] > 3)].sort_values(
                        by='rating', ascending=False)
                    for _, row in user_top_products.iterrows():
                        if row['asin'] not in seen_products and row['asin'] not in test_user_rated_products:
                            recommended_products.append(row)
                            seen_products.add(row['asin'])
                        if len(recommended_products) >= 10:
                            break
                    if len(recommended_products) >= 10:
                        break

                recommended_products_df = pd.DataFrame(recommended_products)
                st.write("**10 sản phẩm được gợi ý:**")
                st.dataframe(recommended_products_df[['asin', 'rating']])
                show_success("Gợi ý sản phẩm thành công!")
            else:
                st.error("Dữ liệu không đủ sản phẩm!")

# Phần 7: Trò chuyện với AI
elif section == "Trò chuyện với AI":
    with st.container():
        st.markdown("<h2>💬 Trò chuyện với AI</h2>", unsafe_allow_html=True)

        # Hiển thị lịch sử trò chuyện
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.markdown(f"<div class='stContainer'><strong>Bạn:</strong> {message['content']}</div>",
                            unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='stContainer'><strong>AI:</strong> {message['content']}</div>",
                            unsafe_allow_html=True)

        user_message = st.text_input("Nhập tin nhắn:", placeholder="Hỏi tôi bất cứ điều gì...")
        if st.button("Gửi"):
            if user_message:
                st.session_state.chat_history.append({"role": "user", "content": user_message})
                headers = {
                    "Authorization": f"Bearer 0d7b4c1d68ed3ff0262b60653fe9aec4a7500c2503c5a9cf03049cff2300e8c4",
                    "Content-Type": "application/json"
                }
                payload = {
                    "model": "meta-llama/Llama-3-8b-chat-hf",
                    "messages": [
                        {"role": "system", "content": "Bạn là một AI hữu ích, trả lời bằng tiếng Việt."},
                        {"role": "user", "content": user_message}
                    ],
                    "max_tokens": 500,
                    "temperature": 0.7
                }
                try:
                    response = requests.post("https://api.together.xyz/v1/chat/completions", headers=headers,
                                             json=payload)
                    response.raise_for_status()
                    bot_response = response.json()["choices"][0]["message"]["content"]
                except Exception as e:
                    bot_response = f"Đã có lỗi: {str(e)}."
                st.session_state.chat_history.append({"role": "assistant", "content": bot_response})
                show_success("Đã nhận phản hồi từ AI!")
                st.experimental_rerun()

        if st.button("Xóa lịch sử trò chuyện"):
            st.session_state.chat_history = []
            show_success("Lịch sử trò chuyện đã được xóa!")
            st.experimental_rerun()

# Footer
st.markdown("""
    <hr>
    <p class='center-text'>© 2025 Nguyễn Trung Hiếu - Ứng dụng NLP</p>
""", unsafe_allow_html=True)