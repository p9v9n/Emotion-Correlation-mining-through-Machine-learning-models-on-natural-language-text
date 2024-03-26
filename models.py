from django.db import models

# Create your models here.
from django.db.models import CASCADE


class ClientRegister_Model(models.Model):
    username = models.CharField(max_length=30)
    email = models.EmailField(max_length=30)
    password = models.CharField(max_length=10)
    phoneno = models.CharField(max_length=10)
    country = models.CharField(max_length=30)
    state = models.CharField(max_length=30)
    city = models.CharField(max_length=30)

class Tweet_Message_details(models.Model):

    Tweet_Id=models.CharField(max_length=300)
    Tweet_Label=models.CharField(max_length=300)
    Tweet_Message=models.CharField(max_length=300)


class Emotion_prediction(models.Model):

    Tweet_Id=models.CharField(max_length=300)
    Tweet_Message=models.CharField(max_length=30000)
    Emotion_Prediction=models.CharField(max_length=300)


class detection_accuracy(models.Model):

    names = models.CharField(max_length=300)
    ratio = models.CharField(max_length=300)

class detection_ratio(models.Model):

    names = models.CharField(max_length=300)
    ratio = models.CharField(max_length=300)



