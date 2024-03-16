from django.shortcuts import render
from django.http import HttpResponseRedirect
from .forms import UploadFileForm

from detection_app.utils import handle_uploaded_file
# Create your views here.

def home(request):
    if request.method == "POST":
        form = UploadFileForm(request.POST, request.FILES)
        # print(form)
        if form.is_valid():
            print("success")
            handle_uploaded_file(request.FILES["video"])
            return render(request,'home.html', {'form':form})
    else:
        form = UploadFileForm()
    return render(request,'home.html', {'form':form})
