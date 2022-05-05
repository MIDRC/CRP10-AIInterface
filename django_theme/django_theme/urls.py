from django.contrib import admin
from django.conf.urls import include, url
from django.urls import path


urlpatterns = [
    path(r'admin/', admin.site.urls),
    path(r'', include('app.urls')), # our application !
    path('', include('django.contrib.auth.urls')),
]