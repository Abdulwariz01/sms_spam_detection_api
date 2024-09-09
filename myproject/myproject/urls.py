from django.contrib import admin
from django.urls import path, include
from cnn_api.views import home

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/', include('cnn_api.urls')),  # Including cnn_api URLs under the 'api/' prefix
    path('', home, name='home'),  # Root path
]
