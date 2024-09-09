from django.urls import path
from .views import predict_view, home

urlpatterns = [
    path('', home, name='home'),  # Root URL of the app
    path('predict/', predict_view, name='predict'),  # Prediction URL
]



