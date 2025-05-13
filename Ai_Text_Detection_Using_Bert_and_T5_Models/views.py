# from django.shortcuts import render, HttpResponse
# from django.contrib import messages
# from users.models import UserRegistrationModel

from django.shortcuts import render


def index(request):
    return render(request, 'index.html', {})

def AdminLogin(request):
    return render(request, 'AdminLogin.html', {})

def UserLogin(request):
    return render(request, 'UserLogin.html', {})


def UserRegisterForm(request):
    return render(request, 'UserRegistrations.html',{}) 







































"""def index(request):
    return render(request, 'index.html')

def AdminLogin(request):
    if request.method == 'POST':
        usrid = request.POST.get('loginid')
        pswd = request.POST.get('pswd')
        print("User ID is = ", usrid)
        if usrid == 'admin' and pswd == 'admin':
            return render(request, 'admins/AdminHome.html')

        else:
            messages.success(request, 'Please Check Your Login Details')
    return render(request, 'AdminLogin.html', {})

def base(request):
    return render(request, 'base.html')

def UserLogin(request):
    return render(request, 'UserLogin.html')

def UserRegistrations(request):
    return render(request, 'UserRegistrations.html')



def RegisterUsersView(request):
    data = UserRegistrationModel.objects.all()
    return render(request, 'admins/viewregisterusers.html', {'data': data})

"""

