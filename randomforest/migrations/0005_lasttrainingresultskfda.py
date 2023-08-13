# Generated by Django 4.2.3 on 2023-08-12 18:57

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('randomforest', '0004_lasttrainingresultsrandomforest_accuracy'),
    ]

    operations = [
        migrations.CreateModel(
            name='LastTrainingResultsKFDA',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('time', models.DateTimeField(auto_now_add=True)),
                ('accuracy', models.CharField(max_length=500)),
                ('roc_curve_image', models.ImageField(blank=True, null=True, upload_to='roc_curves/')),
            ],
        ),
    ]
