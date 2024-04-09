from django.shortcuts import render, redirect
from django.http import HttpResponseRedirect,JsonResponse
from .forms import UploadFileForm
from .data import advocates as adv,districts
from detection_app.utils import handle_uploaded_file, mail
# Create your views here.

def home(request):
    if request.method == "POST":
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            print("success")
            result = handle_uploaded_file(request.FILES["video"])
            # result = 'success'
            return render(request,'result.html', {'result':result})

    else:
        form = UploadFileForm()
    return render(request,'home.html', {'form':form})

def result(request):
    action = request.GET.get('action')
    if action=="report":
        data = {}

        data={
            'name':request.GET.get('name'),
            'path' :  request.GET.get('path'),
            'pnum' :  request.GET.get('pnum'),
            'email' :  request.GET.get('email'),
            'complaint' :  request.GET.get('complaint'),
        }

         
        mail(data)
        return JsonResponse({'success':True})
    return render(request,'result.html')

def advocates(request):
    return render(request,'advocates.html',{'advocates':adv,'districts':districts})

