from django.contrib import admin
from django.urls import path, include
from django.conf.urls import url
from app.views import Home,Registration
from django.conf.urls.static import static
from django.conf import settings


urlpatterns = [
    #path('admin/', admin.site.urls),
    path('', Registration.users, name='users'),
    #path('', include('django.contrib.auth.urls')),

    path('secret/', Registration.secret_page, name='secret'),
    path('signup', Registration.signup, name='signup'),

    #path('index/', Home.index, name='index'),
    path('login_base', Home.login_base, name='login_base'),
    path('index', Home.index, name='index'),
    #url('table',Home.table, name='table'),
    url('Multimodality',Home.loadData, name='Multimodality'),
    url('shapley_value',Home.shapley_values, name='shapley'),
    url('training', Home.training_model, name='training'),
    url('testing',Home.testing, name='testing'),
    url('heat_maps',Home.heat_maps, name='heat_maps'),
    url('activation_maps',Home.activation_maps, name='activation_maps'),

    url(r'run', Home.run, name='run'),
    url(r'^monitor/$', Home.monitor, name='monitor'),
    url(r'^delete_job/(?P<task_id>.+)/$', Home.delete_job,
       name='delete_job'),
    url(r'^cancel_job/(?P<task_id>.+)/$', Home.cancel_job,
       name='cancel_job')
]


urlpatterns += static(settings.MEDIA_URL,document_root=settings.MEDIA_ROOT)
urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
