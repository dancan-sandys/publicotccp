# your_app_name/forms.py
from django import forms
from .models import UploadedFolder
from multiupload.fields import MultiFileField

class FolderInput(forms.FileInput):
    def render(self, name, value, attrs=None, renderer=None):
        attrs['webkitdirectory'] = ''
        attrs['directory'] = ''
        return super().render(name, value, attrs=attrs, renderer=renderer)

class UploadFolderForm(forms.ModelForm):
    folder = forms.FileField(widget=FolderInput)

    class Meta:
        model = UploadedFolder
        fields = ('folder','cellline',)
