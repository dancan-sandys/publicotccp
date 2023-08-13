# your_app_name/views.py
import os
import glob
import shutil
from django.conf import settings
from django.shortcuts import render
from django.http import HttpResponse
from randomforest.models import *
from .forms import UploadFolderForm
from .importdata import parameter_import, parameter_calc

def upload_files_view(request):
    if request.method == 'POST':
        
        print("Check Uploads")
        # print(request.FILES)
        # print(request.FILES.getlist('folder'))

        print("Check Uploads")

        form = UploadFolderForm(request.POST, request.FILES)
        # form.save()
        if form.is_valid():
            # Create a temporary directory to store uploaded files
            temp_dir = os.path.join(settings.MEDIA_ROOT, 'temp_folder')
            os.makedirs(temp_dir, exist_ok=True)

            for file in request.FILES.getlist('folder'):
                uploadfile = UploadedFolder(
                    folder = file,
                )
              
                uploadfile.save()
                
                print("File '{}' copied successfully.".format(file.name))

              
                
            # [X_new, X_labels, objectID, cell_list] = parameter_import()
            # parameter_calc(X_new,objectID)

         

            # Clean up: remove temporary directory and files
            # shutil.rmtree(temp_dir)

    else:
        form = UploadFolderForm()
    
    lastRfTrainingModel =  LastTrainingResultsRandomForest.objects.last()
    
    lastkfdaTrainingModel =  LastTrainingResultsKFDA.objects.last()

    return render(request, 'dataupload.html', {'form': form,'randomForest': lastRfTrainingModel,'kfda': lastkfdaTrainingModel})




def upload_folder_view(request):
    if request.method == 'POST':
        form = UploadFolderForm(request.POST, request.FILES)
        if form.is_valid():
            # Process the uploaded files here
            files = request.FILES.getlist('folder')
            # Call your functions to process the Excel sheets here
            # ...
            return render(request, 'upload_success.html')
    else:
        form = UploadFolderForm()

    return render(request, 'dataupload.html', {'form': form})



def custom_function(request):
    print("starting ")
    if request.method == "POST":
        print("running")
        # Your custom function logic goes here
        # For example, you can perform database operations or any other task
        parameter_import()

        # return render(request, 'dataupload.html', {'form': form})
        return HttpResponse("Function executed successfully")
    return HttpResponse("Invalid request method")