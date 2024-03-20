from django.shortcuts import render, redirect
from django.http import HttpResponseRedirect
from .forms import UploadFileForm

from detection_app.utils import handle_uploaded_file
# Create your views here.

def home(request):
    if request.method == "POST":
        form = UploadFileForm(request.POST, request.FILES)
        print(form)
        if form.is_valid():
            print("success")
            result = handle_uploaded_file(request.FILES["video"])
            # result = 'success'
            return render(request,'result.html', {'result':result})

    else:
        form = UploadFileForm()
    return render(request,'home.html', {'form':form})

def result(request):
    return render(request,'result.html')

