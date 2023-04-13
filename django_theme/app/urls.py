# urls.py file would contain all the URLs which would re-direct to the respective html pages

from django.contrib import admin
from django.urls import path, include
from django.conf.urls import url
from app.views import Home,Registration
from django.conf.urls.static import static
from django.conf import settings

urlpatterns = [
    path('', Registration.users, name='users'),
    path('secret/', Registration.secret_page, name='secret'),
    path('signup', Registration.signup, name='signup'),

    path('login_base', Home.login_base, name='login_base'),
    path('index', Home.index, name='index'),
    url('ui-database',Home.loadData, name='ui-database'),
    url('shapley_value',Home.shapley_values, name='shapley'),
    url('training',Home.run_training,name ='training'),
    url('monitor',Home.monitor_training,name ='monitor_training'),
    url('plot_acc',Home.plot_acc,name ='plot_acc'),
    url('shapley_values',Home.shapley_values,name ='shapley_values'),
    url('testing',Home.testing, name='testing'),
    url('heat_maps',Home.heat_maps, name='heat_maps'),
    url('jupyter_notebook',Home.jupyter_notebook,name='jupyter_notebook'),
    url('activation_maps',Home.activation_maps, name='activation_maps'),
    url('interactive_dropdowns',Home.interactive_dropdowns, name='interactive_dropdowns'),

    url(r'run', Home.run, name='run'),
    url(r'^delete_job/(?P<task_id>.+)/$', Home.delete_job,
       name='delete_job'),
    url(r'^cancel_job/(?P<task_id>.+)/$', Home.cancel_job,
       name='cancel_job')
]


urlpatterns += static(settings.MEDIA_URL,document_root=settings.MEDIA_ROOT)
urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
