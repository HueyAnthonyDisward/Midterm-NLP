from django.contrib import admin
from django.urls import path
from nlp_app import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.preprocessing, name='preprocessing'),
    path('preprocessing/', views.preprocessing, name='preprocessing'),
    path('modeling/', views.modeling, name='modeling'),
    path('modeling/train/', views.train_model, name='train_model'),
    path('modeling/predict/', views.predict_text, name='predict_text'),
    path('recommendation/', views.recommendation, name='recommendation'),
    path('chatbot/', views.chatbot, name='chatbot'),
    path('crawl/', views.crawl_data, name='crawl_data'),
    path('augment/', views.augment_data, name='augment_data'),
    path('preprocess/', views.preprocess_text, name='preprocess_text'),
    path('model/', views.model_text, name='model_text'),
    path('collaborative_filtering/', views.collaborative_filtering, name='collaborative_filtering'),
    path('content_based_filtering/', views.content_based_filtering, name='content_based_filtering'),
    path('context_based_filtering/', views.context_based_filtering, name='context_based_filtering'),
    path('chatbot_api/', views.chatbot_api, name='chatbot_api'),
    path('chatbot_trained/', views.chatbot_trained, name='chatbot_trained'),
]