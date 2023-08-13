from django.db import models

# Create your models here.

class CellLines(models.Model):
    name = models.CharField(max_length = 500)
    
    def __str__(self):
        return f"{self.name}"

class UploadedFolder(models.Model):
    folder = models.FileField(upload_to='uploaded_folders/')
    uploaded_at = models.DateTimeField(auto_now_add=True)
    cellline = models.ForeignKey(CellLines,default = 1,on_delete =  models.CASCADE)

    def __str__(self):
        return f"{self.folder.name} - {self.uploaded_at}"

class LastTrainingResultsRandomForest(models.Model):
    time = models.DateTimeField(auto_now_add=True)
    accuracy = models.CharField(max_length = 500)
    roc_curve_image = models.ImageField(upload_to='roc_curves/', null=True, blank=True)
    number_of_cells =  models.CharField(max_length = 500)

    

    # def __str__(self):
    #     return f"{self.folder.name} - {self.uploaded_at}"
    

class LastTrainingResultsKFDA(models.Model):
    time = models.DateTimeField(auto_now_add=True)
    accuracy = models.CharField(max_length = 500)
    roc_curve_image = models.ImageField(upload_to='roc_curves/', null=True, blank=True)
    number_of_cells =  models.CharField(max_length = 500)
    
    

    # def __str__(self):
    #     return f"{self.folder.name} - {self.uploaded_at}"
    

