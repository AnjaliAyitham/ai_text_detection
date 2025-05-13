"""
URL configuration for employee_attrition project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""

from django.contrib import admin
from django.urls import path
from Ai_Text_Detection_Using_Bert_and_T5_Models import views as mainView
from admins import views as admins
from users import views as usr
from django.contrib.staticfiles.urls import static
from django.contrib.staticfiles.urls import staticfiles_urlpatterns
from django.conf import settings
from users import views as uv

urlpatterns = [
    path('admin/', admin.site.urls),
    path("", mainView.index, name="index"),
    path("index", mainView.index, name="index"),
    path("Adminlogin", mainView.AdminLogin, name="AdminLogin"),
    path("UserLogin", mainView.UserLogin, name="UserLogin"),
    path('UserRegisterForm',mainView.UserRegisterForm,name='UserRegisterForm'),

    # admin views
    path("AdminLogincheck", admins.AdminLoginCheck, name="AdminLoginCheck"),
    path('userDetails', admins.RegisterUsersView, name='RegisterUsersView'),
    path('ActivUsers/', admins.ActivaUsers, name='activate_users'),
    path('DeleteUsers/', admins.DeleteUsers, name='delete_users'),
    
    
    #userurls
    path('UserRegister',uv.UserRegisterActions,name='UserRegister'),
    path("UserLoginCheck/", usr.UserLoginCheck, name="UserLoginCheck"),
    path("UserHome/", usr.UserHome, name="UserHome"),
    path("prediction/", usr.prediction, name="prediction"),
    
    path("trainingPage", usr.trainingPage, name="trainingPage"),
    
    path("training", usr.training, name="training"),
    path("index/", usr.index, name="index"),
   

]+ static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)





































"""
from django.contrib import admin
from django.urls import path
from employee_attrition import views as mv

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', mv.index, name='index'),
    
    path('admin_login', mv.AdminLogin, name='admin_login'),
    
    path('UserLogin', mv.UserLogin, name='UserLogin'),
    
    path('User-Registrations', mv.UserRegistrations, name='register'),
    
    path('RegisterUsersView', mv.RegisterUsersView, name='RegisterUsersView'),

]
"""
# 18	1	2	1051	0	0	0	0	0	0	0
# 22	1	1	1606	1	1	0	0	0	0	1



