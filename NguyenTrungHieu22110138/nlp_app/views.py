from datetime import datetime
import os
import base64
import pickle
import joblib
from io import BytesIO
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nlpaug.augmenter.word as naw
import pandas as pd
import json
import numpy as np
from django.http import JsonResponse
from django.shortcuts import render
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.decomposition import TruncatedSVD
from gensim.models import Word2Vec, FastText
from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import requests
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json

def preprocessing(request):
    return render(request, 'preprocessing.html', {'text_data': request.session.get('text_data', '')})

def modeling(request):
    return render(request, 'modeling.html')

def recommendation(request):
    return render(request, 'recommendation.html')

def chatbot(request):
    return render(request, 'chatbot.html')

def crawl_data(request):
    if request.method == 'POST':
        option = request.POST.get('option')
        text_data = request.session.get('text_data', '')
        error = None
        success = None

        if option == 'web':
            url = request.POST.get('url')
            try:
                response = requests.get(url)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, 'html.parser')
                    paragraphs = soup.find_all('p')
                    text_data = " ".join([p.get_text() for p in paragraphs])
                    success = "Cào dữ liệu thành công!"
                else:
                    error = "Không thể cào dữ liệu từ URL!"
            except:
                error = "URL không hợp lệ hoặc lỗi kết nối!"
        
        elif option == 'text':
            text_data = request.POST.get('text_input')
            success = "Văn bản đã được lưu!"
        
        elif option == 'txt':
            file = request.FILES.get('file_txt')
            if file:
                text_data = file.read().decode('utf-8')
                success = "Tải file TXT thành công!"
            else:
                error = "Vui lòng chọn file TXT!"
        
        elif option == 'json':
            file = request.FILES.get('file_json')
            if file:
                data = json.load(file)
                text_data = " ".join([str(item) for item in data.values()])
                success = "Tải file JSON thành công!"
            else:
                error = "Vui lòng chọn file JSON!"
        
        elif option == 'csv':
            file = request.FILES.get('file_csv')
            if file:
                df = pd.read_csv(file)
                text_data = " ".join(df.astype(str).values.flatten())
                success = "Tải file CSV thành công!"
            else:
                error = "Vui lòng chọn file CSV!"

        request.session['text_data'] = text_data
        return JsonResponse({'text_data': text_data[:500], 'success': success, 'error': error, 'saved': True})
    return JsonResponse({'error': 'Phương thức không hợp lệ!'})

def augment_data(request):
    if request.method == 'POST':
        augment_method = request.POST.get('augment_method')
        is_labeled = request.POST.get('is_labeled') == 'true'
        label_column = request.POST.get('label_column', '')
        text_data = request.session.get('text_data', '')
        if not text_data:
            return JsonResponse({'error': 'Chưa có dữ liệu để tăng cường!'})

        synonym_aug = naw.SynonymAug(aug_src='wordnet', aug_p=0.2)
        random_insert_aug = naw.RandomWordAug(action="insert", aug_p=0.2)
        random_swap_aug = naw.RandomWordAug(action="swap", aug_p=0.2)
        random_delete_aug = naw.RandomWordAug(action="delete", aug_p=0.2)

        augmented_text = text_data
        if is_labeled:
            try:
                if text_data.startswith('[{') or text_data.startswith('{'):  # JSON data
                    data = json.loads(text_data)
                    if not label_column or label_column not in data[0]:
                        return JsonResponse({'error': 'Tên cột nhãn không hợp lệ!'})
                    for item in data:
                        text = " ".join([str(v) for k, v in item.items() if k != label_column])
                        if augment_method == "Thay thế từ đồng nghĩa":
                            item['text'] = synonym_aug.augment(text)
                        elif augment_method == "Chèn từ ngẫu nhiên":
                            item['text'] = random_insert_aug.augment(text)
                        elif augment_method == "Đổi chỗ từ":
                            item['text'] = random_swap_aug.augment(text)
                        elif augment_method == "Xóa từ ngẫu nhiên":
                            item['text'] = random_delete_aug.augment(text)
                    augmented_text = json.dumps(data)
                
                elif '\n' in text_data:  # CSV-like data
                    df = pd.read_csv(pd.compat.StringIO(text_data))
                    if not label_column or label_column not in df.columns:
                        return JsonResponse({'error': 'Tên cột nhãn không hợp lệ!'})
                    for i, row in df.iterrows():
                        text = " ".join([str(v) for k, v in row.items() if k != label_column])
                        if augment_method == "Thay thế từ đồng nghĩa":
                            df.at[i, 'text'] = synonym_aug.augment(text)
                        elif augment_method == "Chèn từ ngẫu nhiên":
                            df.at[i, 'text'] = random_insert_aug.augment(text)
                        elif augment_method == "Đổi chỗ từ":
                            df.at[i, 'text'] = random_swap_aug.augment(text)
                        elif augment_method == "Xóa từ ngẫu nhiên":
                            df.at[i, 'text'] = random_delete_aug.augment(text)
                    augmented_text = df.to_csv(index=False)
                
                else:  # Text with tab-separated label
                    text, label = text_data.rsplit('\t', 1)
                    if augment_method == "Thay thế từ đồng nghĩa":
                        augmented_text = synonym_aug.augment(text)
                    elif augment_method == "Chèn từ ngẫu nhiên":
                        augmented_text = random_insert_aug.augment(text)
                    elif augment_method == "Đổi chỗ từ":
                        augmented_text = random_swap_aug.augment(text)
                    elif augment_method == "Xóa từ ngẫu nhiên":
                        augmented_text = random_delete_aug.augment(text)
                    augmented_text = f"{augmented_text}\t{label}"
            except Exception as e:
                return JsonResponse({'error': f'Dữ liệu có nhãn không đúng định dạng! Lỗi: {str(e)}'})
        else:
            if augment_method == "Thay thế từ đồng nghĩa":
                augmented_text = synonym_aug.augment(text_data)
            elif augment_method == "Chèn từ ngẫu nhiên":
                augmented_text = random_insert_aug.augment(text_data)
            elif augment_method == "Đổi chỗ từ":
                augmented_text = random_swap_aug.augment(text_data)
            elif augment_method == "Xóa từ ngẫu nhiên":
                augmented_text = random_delete_aug.augment(text_data)

        request.session['augmented_text'] = augmented_text
        save = request.POST.get('save')
        if save == 'true':
            request.session['text_data'] = augmented_text
            return JsonResponse({'augmented_text': augmented_text, 'success': 'Kết quả đã được lưu vào dữ liệu chính!'})
        return JsonResponse({'augmented_text': augmented_text, 'success': 'Tăng cường dữ liệu thành công!'})
    return JsonResponse({'error': 'Phương thức không hợp lệ!'})

def preprocess_text(request):
    if request.method == 'POST':
        option_process = request.POST.get('option_process')
        text_data = request.session.get('text_data', '')
        if not text_data:
            return JsonResponse({'error': 'Chưa có dữ liệu để xử lý!'})

        processed_text = ''
        if option_process == "Cleaning":
            processed_text = text_data.replace(".", "").replace(",", "")
        elif option_process == "Tokenization":
            processed_text = word_tokenize(text_data)
        elif option_process == "Stopwords Removal":
            stop_words = set(stopwords.words("english"))
            words = word_tokenize(text_data)
            processed_text = [word for word in words if word.lower() not in stop_words]
        elif option_process == "Lemmatization":
            lemmatizer = WordNetLemmatizer()
            words = word_tokenize(text_data)
            processed_text = [lemmatizer.lemmatize(word) for word in words]

        request.session['processed_text'] = processed_text
        save = request.POST.get('save')
        if save == 'true':
            request.session['text_data'] = " ".join(processed_text) if isinstance(processed_text, list) else processed_text
            return JsonResponse({'processed_text': processed_text, 'success': 'Kết quả đã được lưu!'})
        return JsonResponse({'processed_text': processed_text, 'success': 'Xử lý văn bản thành công!'})
    return JsonResponse({'error': 'Phương thức không hợp lệ!'})

def model_text(request):
    if request.method == 'POST':
        method = request.POST.get('method')
        i = int(request.POST.get('i'))
        j = int(request.POST.get('j'))
        text_data = request.session.get('text_data', '')
        if not text_data:
            return JsonResponse({'error': 'Chưa có dữ liệu để xử lý!'})

        words = text_data.split()
        selected_text = " ".join(words[i-1:j])
        result = []

        if method == "One-Hot Encoding":
            encoder = LabelBinarizer()
            selected_words = selected_text.split()
            encoded = encoder.fit_transform(selected_words)
            for word, encoding in zip(selected_words, encoded):
                result.append(f"{word}: {list(encoding)}")

        elif method == "Bag of Words":
            vectorizer = CountVectorizer()
            bow_matrix = vectorizer.fit_transform([selected_text])
            words = vectorizer.get_feature_names_out()
            counts = bow_matrix.toarray()[0]
            for word, count in zip(words, counts):
                result.append(f"{word}: {count}")

        elif method == "TF-IDF":
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform([selected_text])
            words = vectorizer.get_feature_names_out()
            scores = tfidf_matrix.toarray()[0]
            for word, score in zip(words, scores):
                result.append(f"{word}: {score:.4f}")

        elif method == "Bag of N-Gram":
            vectorizer = CountVectorizer(ngram_range=(3, 3))
            ngram_matrix = vectorizer.fit_transform([selected_text])
            ngrams = vectorizer.get_feature_names_out()
            counts = ngram_matrix.toarray()[0]
            for ngram, count in zip(ngrams, counts):
                result.append(f"{ngram}: {count}")

        elif method in ["Word2Vec", "CBOW"]:
            words_tokenized = [word_tokenize(selected_text.lower())]
            sg_value = 1 if method == "Word2Vec" else 0
            model = Word2Vec(words_tokenized, vector_size=100, window=5, min_count=1, workers=4, sg=sg_value)
            for word in words_tokenized[0]:
                result.append(f"{word}: {model.wv[word].tolist()[:5]}...")

        elif method == "BERT":
            tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
            model = BertModel.from_pretrained("bert-base-uncased")
            tokens = tokenizer(selected_text, return_tensors="pt", padding=True, truncation=True)
            with torch.no_grad():
                outputs = model(**tokens)
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze().tolist()
            result.append(f"Vector BERT: {embedding[:5]}")

        elif method == "FastText":
            words_tokenized = [word_tokenize(selected_text.lower())]
            model = FastText(words_tokenized, vector_size=100, window=5, min_count=1, workers=4)
            for word in words_tokenized[0]:
                result.append(f"{word}: {model.wv[word].tolist()[:5]}...")

        return JsonResponse({'result': result, 'success': 'Chuyển đổi thành công!'})
    return JsonResponse({'error': 'Phương thức không hợp lệ!'})

# Huấn luyện mô hình


def train_model(request):
    if request.method == 'POST':
        vectorization_method = request.POST.get('vectorization_method')
        model_type = request.POST.get('model_type')
        label_column = request.POST.get('label_column')
        csv_file = request.FILES.get('csv_file')

        if not csv_file or not label_column:
            return JsonResponse({'error': 'Vui lòng tải file CSV và nhập tên cột nhãn!'})

        try:
            df = pd.read_csv(csv_file)
            if label_column not in df.columns or df.shape[1] < 2:
                return JsonResponse({'error': 'File CSV không chứa cột nhãn hoặc dữ liệu không hợp lệ!'})

            text_data = df.iloc[:, 0].astype(str)
            labels = df[label_column]
            label_encoder = LabelEncoder()
            labels_encoded = label_encoder.fit_transform(labels)

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
                vectorizer = CountVectorizer(ngram_range=(3, 3), max_features=5000)
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

            if X is None or X.shape[0] == 0:
                return JsonResponse({'error': 'Dữ liệu không hợp lệ sau khi vector hóa!'})

            if isinstance(X, np.ndarray) and X.ndim == 1:
                X = X.reshape(-1, 1)
            if isinstance(X, np.ndarray) and X.shape[1] > 300:
                svd = TruncatedSVD(n_components=300)
                X = svd.fit_transform(X)

            X_train, X_test, y_train, y_test = train_test_split(X, labels_encoded, test_size=0.2, random_state=42)

            model = None
            if model_type == "Naive Bayes":
                if vectorization_method in ["Word2Vec", "CBOW"]:
                    return JsonResponse({'error': 'MultinomialNB không hỗ trợ Word2Vec hoặc CBOW!'})
                model = MultinomialNB()
            elif model_type == "SVM":
                model = SVC(probability=True)
            elif model_type == "Logistic Regression":
                model = LogisticRegression()
            elif model_type == "KNN":
                k = int(request.POST.get('knn_k', 5))
                model = KNeighborsClassifier(n_neighbors=k)
            elif model_type == "Decision Tree":
                depth = int(request.POST.get('dt_depth', 10))
                model = DecisionTreeClassifier(max_depth=depth)
            elif model_type == "Random Forest":
                model = RandomForestClassifier(random_state=42)
            elif model_type == "Gradient Boosting":
                model = GradientBoostingClassifier(random_state=42)
            elif model_type == "XGBoost":
                model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
            elif model_type == "AdaBoost":
                model = AdaBoostClassifier(random_state=42)

            if model:
                # Debug: Kiểm tra trước khi lưu mô hình
                try:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                except Exception as e:
                    return JsonResponse({'error': f'Lỗi khi huấn luyện mô hình: {str(e)}'})

                accuracy = accuracy_score(y_test, y_pred)
                report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
                matrix = confusion_matrix(y_test, y_pred)

                # Tạo ma trận nhầm lẫn dưới dạng hình ảnh
                plt.figure(figsize=(6, 4))
                sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
                plt.title(f'Ma trận nhầm lẫn ({model_type}, {vectorization_method})')
                plt.xlabel('Dự đoán')
                plt.ylabel('Thực tế')
                buffer = BytesIO()
                plt.savefig(buffer, format='png', bbox_inches='tight')
                plt.close()
                matrix_image = base64.b64encode(buffer.getvalue()).decode('utf-8')

                # Lưu mô hình và vectorizer vào file tạm thời
                model_dir = os.path.join(os.path.dirname(__file__), 'temp_models')
                os.makedirs(model_dir, exist_ok=True)
                model_key = f"{model_type}_{vectorization_method}_{len(request.session.get('model_results', {}))}"
                model_path = os.path.join(model_dir, f"{model_key}_model.joblib")
                vectorizer_path = os.path.join(model_dir, f"{model_key}_vectorizer.joblib")
                label_encoder_path = os.path.join(model_dir, f"{model_key}_label_encoder.joblib")

                # Debug: Kiểm tra lưu mô hình
                try:
                    joblib.dump(model, model_path)
                    joblib.dump(vectorizer, vectorizer_path)
                    joblib.dump(label_encoder, label_encoder_path)
                except Exception as e:
                    return JsonResponse({'error': f'Lỗi khi lưu mô hình: {str(e)}'})

                # Lưu kết quả vào session
                if 'model_results' not in request.session:
                    request.session['model_results'] = {}
                request.session['model_results'][model_key] = {
                    'accuracy': accuracy,
                    'matrix_image': matrix_image,
                    'report': report,
                    'vectorization_method': vectorization_method,
                    'model_type': model_type
                }
                request.session['current_model'] = {
                    'model_path': model_path,
                    'vectorizer_path': vectorizer_path,
                    'label_encoder_path': label_encoder_path,
                    'vectorization_method': vectorization_method
                }
                request.session.modified = True

                return JsonResponse({
                    'success': 'Mô hình đã được huấn luyện thành công!',
                    'accuracy': accuracy,
                    'report': report,
                    'matrix_image': matrix_image,
                    'model_results': request.session['model_results']
                })
            return JsonResponse({'error': 'Không thể huấn luyện mô hình!'})
        except Exception as e:
            return JsonResponse({'error': f'Lỗi khi xử lý file CSV: {str(e)}'})
    return JsonResponse({'error': 'Phương thức không hợp lệ!'})

def predict_text(request):
    if request.method == 'POST':
        text_input = request.POST.get('text_input')
        if not text_input:
            return JsonResponse({'error': 'Vui lòng nhập văn bản để dự đoán!'})

        if 'current_model' not in request.session:
            return JsonResponse({'error': 'Chưa có mô hình nào được huấn luyện!'})

        current_model = request.session['current_model']
        try:
            model = joblib.load(current_model['model_path'])
            vectorizer = joblib.load(current_model['vectorizer_path'])
            label_encoder = joblib.load(current_model['label_encoder_path'])
        except Exception as e:
            return JsonResponse({'error': f'Lỗi khi tải mô hình: {str(e)}'})

        vectorization_method = current_model['vectorization_method']

        input_vector = None
        if vectorization_method == "One-Hot Encoding":
            input_vector = vectorizer.transform([text_input])
        elif vectorization_method == "Bag of Words":
            input_vector = vectorizer.transform([text_input])
        elif vectorization_method == "TF-IDF":
            input_vector = vectorizer.transform([text_input])
        elif vectorization_method == "Bag of N-Gram":
            input_vector = vectorizer.transform([text_input])
        elif vectorization_method in ["Word2Vec", "CBOW"]:
            tokenized_input = word_tokenize(text_input.lower())
            input_vector = np.mean([vectorizer.wv[word] for word in tokenized_input if word in vectorizer.wv], axis=0)
            if input_vector.shape == (100,):
                input_vector = input_vector.reshape(1, -1)
            else:
                input_vector = np.zeros((1, 100))

        if input_vector is not None:
            try:
                prediction = model.predict(input_vector)
                prediction_label = label_encoder.inverse_transform(prediction)[0]
            except Exception as e:
                return JsonResponse({'error': f'Lỗi khi dự đoán: {str(e)}'})
            return JsonResponse({
                'success': 'Dự đoán thành công!',
                'prediction': prediction_label
            })
        return JsonResponse({'error': 'Không thể chuyển đổi văn bản!'})
    return JsonResponse({'error': 'Phương thức không hợp lệ!'})


#Hệ khuyến nghị
def collaborative_filtering(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'Phương thức không hợp lệ!'}, status=405)

    csv_file = request.FILES.get('csv_file')
    user_col = request.POST.get('user_col')
    item_col = request.POST.get('item_col')
    rating_col = request.POST.get('rating_col')
    target_user = request.POST.get('target_user')
    method = request.POST.get('method')  # 'user' or 'item'

    if not csv_file or not user_col or not item_col or not rating_col or not target_user:
        return JsonResponse({'error': 'Vui lòng cung cấp đầy đủ file CSV và tên cột!'}, status=400)

    # Đường dẫn đến file ánh xạ (thư mục Data cùng cấp với manage.py)
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    USER_MAPPING_PATH = os.path.join(BASE_DIR, 'Data', 'user_mapping.csv')
    ITEM_MAPPING_PATH = os.path.join(BASE_DIR, 'Data', 'item_mapping.csv')

    try:
        # Đọc file ánh xạ
        user_mapping_df = pd.read_csv(USER_MAPPING_PATH)
        item_mapping_df = pd.read_csv(ITEM_MAPPING_PATH)
        user_mapping = dict(zip(user_mapping_df['userId'], user_mapping_df['name']))
        item_mapping = dict(zip(item_mapping_df['itemId'], item_mapping_df['asin']))
    except FileNotFoundError as e:
        return JsonResponse({'error': f'Không tìm thấy file ánh xạ: {str(e)}'}, status=500)
    except Exception as e:
        return JsonResponse({'error': f'Lỗi khi đọc file ánh xạ: {str(e)}'}, status=500)

    try:
        # Đọc file CSV
        df = pd.read_csv(csv_file)
        if not all(col in df.columns for col in [user_col, item_col, rating_col]):
            return JsonResponse({'error': f'Tên cột không hợp lệ! Cần: {user_col}, {item_col}, {rating_col}'}, status=400)

        # Kiểm tra dữ liệu
        if df[[user_col, item_col, rating_col]].isnull().any().any():
            return JsonResponse({'error': 'Dữ liệu CSV chứa giá trị null!'}, status=400)

        # Tạo ma trận đánh giá
        try:
            rating_matrix = df.pivot(index=user_col, columns=item_col, values=rating_col).fillna(0)
        except Exception as e:
            return JsonResponse({'error': f'Lỗi khi tạo ma trận đánh giá: {str(e)}'}, status=400)

        user_ids = rating_matrix.index

        # Kiểm tra target_user
        try:
            target_user = int(target_user)
        except ValueError:
            return JsonResponse({'error': 'userId phải là một số nguyên!'}, status=400)
        if user_ids.dtype == object:
            target_user = str(target_user)
        if target_user not in user_ids:
            return JsonResponse({'error': f'Người dùng {target_user} không tồn tại trong dữ liệu!'}, status=400)

        # Lấy name của target_user
        target_user_name = user_mapping.get(target_user, f"Unknown User {target_user}")

        # Sử dụng SVD cho Matrix Factorization
        try:
            n_components = min(50, min(rating_matrix.shape) - 1, rating_matrix.shape[1] - 1)
            if n_components < 1:
                return JsonResponse({'error': 'Dữ liệu quá ít để áp dụng SVD! Cần ít nhất 2 item.'}, status=400)
            svd = TruncatedSVD(n_components=n_components, random_state=42)
            matrix_reduced = svd.fit_transform(rating_matrix)
            user_factors = pd.DataFrame(matrix_reduced, index=user_ids)
            item_factors = pd.DataFrame(svd.components_.T, index=rating_matrix.columns)
        except Exception as e:
            return JsonResponse({'error': f'Lỗi khi áp dụng SVD: {str(e)}'}, status=500)

        if method == 'user':
            # Tính độ tương đồng giữa user
            try:
                user_sim = cosine_similarity(user_factors)
                sim_df = pd.DataFrame(user_sim, index=user_ids, columns=user_ids)
                similar_users = sim_df[target_user].sort_values(ascending=False)[1:5]
                
                # Đề xuất item từ user tương tự
                recommendations = []
                for user in similar_users.index:
                    user_ratings = rating_matrix.loc[user]
                    top_items = user_ratings[user_ratings > 0].sort_values(ascending=False).index[:5]
                    recommendations.extend(top_items)
                recommendations = list(dict.fromkeys(recommendations))[:5]

            except Exception as e:
                return JsonResponse({'error': f'Lỗi khi tính tương đồng user: {str(e)}'}, status=500)

        elif method == 'item':
            # Tính độ tương đồng giữa item
            try:
                if item_factors.shape[0] != len(rating_matrix.columns):
                    return JsonResponse({'error': 'Kích thước item_factors không khớp với số item!'}, status=500)
                item_sim = cosine_similarity(item_factors)
                sim_df = pd.DataFrame(item_sim, index=rating_matrix.columns, columns=rating_matrix.columns)
                user_ratings = rating_matrix.loc[target_user]
                rated_items = user_ratings[user_ratings > 0].index
                
                recommendations = []
                for item in rated_items:
                    similar_items = sim_df[item].sort_values(ascending=False)[1:5]
                    recommendations.extend(similar_items.index)
                recommendations = list(dict.fromkeys(recommendations))[:5]

            except Exception as e:
                return JsonResponse({'error': f'Lỗi khi tính tương đồng item: {str(e)}'}, status=500)

        else:
            return JsonResponse({'error': 'Phương thức không hợp lệ! Chọn "user" hoặc "item"'}, status=400)

        # Chuyển itemId thành asin
        recommendations_asin = [item_mapping.get(item, f"Unknown Item {item}") for item in recommendations]

        return JsonResponse({
            'success': 'Khuyến nghị thành công với SVD!',
            'user_name': target_user_name,
            'recommendations': recommendations_asin
        }, status=200)

    except Exception as e:
        return JsonResponse({'error': f'Lỗi khi xử lý file CSV: {str(e)}'}, status=500)
    
def content_based_filtering(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'Phương thức không hợp lệ!'}, status=405)

    csv_file = request.FILES.get('csv_file')
    target_movie_id = request.POST.get('target_movie_id')

    if not csv_file or not target_movie_id:
        return JsonResponse({'error': 'Vui lòng cung cấp file CSV và movieId!'}, status=400)

    try:
        # Đọc file CSV
        df = pd.read_csv(csv_file)
        if 'movieId' not in df.columns or 'title' not in df.columns or 'genres' not in df.columns:
            return JsonResponse({'error': 'File CSV phải có cột movieId, title, và genres!'}, status=400)

        # Kiểm tra dữ liệu null
        if df[['movieId', 'title', 'genres']].isnull().any().any():
            return JsonResponse({'error': 'Dữ liệu CSV chứa giá trị null!'}, status=400)

        # Chuyển đổi genres thành vector đặc trưng sử dụng TF-IDF
        tfidf = TfidfVectorizer(tokenizer=lambda x: x.split('|'))
        genre_matrix = tfidf.fit_transform(df['genres'])

        # Chuyển target_movie_id thành số nguyên
        try:
            target_movie_id = int(target_movie_id)
        except ValueError:
            return JsonResponse({'error': 'movieId phải là một số nguyên!'}, status=400)
        if target_movie_id not in df['movieId'].values:
            return JsonResponse({'error': f'Phim với movieId {target_movie_id} không tồn tại trong dữ liệu!'}, status=400)

        # Lấy index của phim mục tiêu
        target_idx = df.index[df['movieId'] == target_movie_id].tolist()[0]

        # Tính độ tương đồng cosine giữa phim mục tiêu và tất cả phim khác
        similarity_scores = cosine_similarity(genre_matrix[target_idx], genre_matrix)[0]

        # Lấy các phim khác (không phải phim mục tiêu)
        similar_movie_indices = np.argsort(similarity_scores)[::-1][1:6]  # Top 5 phim tương tự, bỏ qua chính nó

        # Lấy movieId và title của các phim được khuyến nghị
        recommended_movies = df.iloc[similar_movie_indices][['movieId', 'title']].values.tolist()

        return JsonResponse({
            'success': 'Khuyến nghị thành công với Content-based Filtering!',
            'target_movie': df[df['movieId'] == target_movie_id]['title'].values[0],
            'recommendations': [{'movieId': movie[0], 'title': movie[1]} for movie in recommended_movies]
        }, status=200)

    except Exception as e:
        return JsonResponse({'error': f'Lỗi khi xử lý: {str(e)}'}, status=500)

def context_based_filtering(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'Phương thức không hợp lệ!'}, status=405)

    csv_file = request.FILES.get('csv_file')
    item_col = request.POST.get('item_col')
    content_col = request.POST.get('content_col')
    context_col = request.POST.get('context_col')
    target_item = request.POST.get('target_item')
    context_value = request.POST.get('context_value')

    if not csv_file or not item_col or not content_col or not context_col or not target_item or not context_value:
        return JsonResponse({'error': 'Vui lòng cung cấp đầy đủ file CSV và thông tin!'}, status=400)

    try:
        # Đọc file CSV
        df = pd.read_csv(csv_file)
        if item_col not in df.columns or content_col not in df.columns or context_col not in df.columns:
            return JsonResponse({'error': 'Tên cột không hợp lệ!'}, status=400)

        # Chuyển đổi context_col (reviews.date) thành datetime và trích xuất ngày trong tuần
        df[context_col] = pd.to_datetime(df[context_col], errors='coerce')
        df['weekday'] = df[context_col].apply(lambda x: x.weekday() if pd.notnull(x) else None)

        # Lọc theo ngữ cảnh dựa trên ngày trong tuần
        # context_value: 'weekday' (0-4) hoặc 'weekend' (5-6)
        if context_value == 'weekday':
            df = df[df['weekday'].notna() & (df['weekday'] < 5)]
        elif context_value == 'weekend':
            df = df[df['weekday'].notna() & (df['weekday'] >= 5)]
        else:
            return JsonResponse({'error': 'context_value phải là "weekday" hoặc "weekend"!'}, status=400)

        if df.empty:
            return JsonResponse({'error': 'Không có dữ liệu phù hợp với ngữ cảnh!'}, status=400)

        # Trích xuất đặc trưng bằng TF-IDF
        tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
        tfidf_matrix = tfidf.fit_transform(df[content_col].fillna(''))

        # Tìm index của target_item
        if target_item not in df[item_col].values:
            return JsonResponse({'error': f'Item {target_item} không tồn tại!'}, status=400)
        target_idx = df[df[item_col] == target_item].index[0]
        target_vector = tfidf_matrix[target_idx]

        # Tính độ tương đồng cosine
        sim_scores = cosine_similarity(target_vector, tfidf_matrix).flatten()

        # Sắp xếp và lấy top 5 item (bỏ target_item)
        top_indices = sim_scores.argsort()[-6:-1][::-1]  # Top 5, bỏ item mục tiêu
        recommendations = df.iloc[top_indices][[item_col, 'name']].drop_duplicates().values.tolist()

        return JsonResponse({
            'success': 'Khuyến nghị thành công với Content-based Filtering!',
            'target_item': target_item,
            'target_name': df[df[item_col] == target_item]['name'].iloc[0],
            'recommendations': [{'asins': item[0], 'name': item[1]} for item in recommendations]
        }, status=200)

    except Exception as e:
        return JsonResponse({'error': f'Lỗi khi xử lý: {str(e)}'}, status=500)
    
@csrf_exempt
def chatbot_api(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'Phương thức không hợp lệ!'}, status=405)

    try:
        data = json.loads(request.body)
        user_message = data.get('message')
        
        if not user_message:
            return JsonResponse({'error': 'Vui lòng cung cấp tin nhắn!'}, status=400)

        # Cấu hình API
        headers = {
            "Authorization": f"Bearer d2bf9ab629851af9f6274fbf9ca9afe2f01671637056709400f6a967b5f42766",
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

        # Gửi yêu cầu đến API
        response = requests.post("https://api.together.xyz/v1/chat/completions", headers=headers, json=payload)
        response.raise_for_status()
        bot_response = response.json()["choices"][0]["message"]["content"]

        return JsonResponse({'success': 'Phản hồi thành công!', 'response': bot_response}, status=200)

    except Exception as e:
        return JsonResponse({'error': f'Lỗi khi xử lý: {str(e)}'}, status=500)

@csrf_exempt
def chatbot_trained(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'Phương thức không hợp lệ!'}, status=405)

    try:
        data = json.loads(request.body)
        user_message = data.get('message')
        
        if not user_message:
            return JsonResponse({'error': 'Vui lòng cung cấp tin nhắn!'}, status=400)

        # Lấy đường dẫn gốc của dự án (thư mục chứa manage.py)
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_dir = os.path.join(base_dir, 'Data')

        # Load các file từ thư mục Data
        with open(os.path.join(data_dir, 'tfidf_vectorizer.pkl'), 'rb') as f:
            tfidf_vectorizer = pickle.load(f)
        with open(os.path.join(data_dir, 'knn_model.pkl'), 'rb') as f:
            knn_model = pickle.load(f)
        with open(os.path.join(data_dir, 'tfidf_matrix.pkl'), 'rb') as f:
            tfidf_matrix = pickle.load(f)
        with open(os.path.join(data_dir, 'answers.pkl'), 'rb') as f:
            answers = pickle.load(f)

        # Chuyển đổi tin nhắn người dùng thành vector TF-IDF
        user_tfidf = tfidf_vectorizer.transform([user_message])

        # Dự đoán chỉ số câu trả lời gần nhất
        nearest_idx = knn_model.kneighbors(user_tfidf, n_neighbors=1, return_distance=False)[0][0]

        # Lấy câu trả lời tương ứng
        bot_response = answers[nearest_idx]

        return JsonResponse({'success': 'Phản hồi thành công!', 'response': bot_response}, status=200)

    except Exception as e:
        return JsonResponse({'error': f'Lỗi khi xử lý: {str(e)}'}, status=500)