import os
from django.core.files.storage import FileSystemStorage
from deepfake_detection.settings import VIDEO_UPLOAD_PATH

import shutil
def handle_uploaded_file(file):
    shutil.rmtree(VIDEO_UPLOAD_PATH)
    if not os.path.exists(VIDEO_UPLOAD_PATH):
        os.makedirs(VIDEO_UPLOAD_PATH)
    fs = FileSystemStorage(VIDEO_UPLOAD_PATH) #defaults to   MEDIA_ROOT  
    filename = fs.save(file.name, file)
    # print('saved')
    file_url = fs.url(filename)
    video_path = VIDEO_UPLOAD_PATH +'/'+ file.name

    prediction = detectFakeVideo(video_path)
    # print(prediction)
    if prediction[0] == 0:
          output = False
    else:
          output = True
    confidence = prediction[1]
    data = {'output': output, 'confidence': round(confidence,3),'path':'uploaded_videos/'+file.name}
    # print(data)
    # os.remove(video_path)
    return data
    




# Used for DL applications, computer vision related processes
import torch
import torchvision

# For image preprocessing
from torchvision import transforms

# Combines dataset & sampler to provide iterable over the dataset
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

import numpy as np
import cv2

# To recognise face from extracted frames
import face_recognition

# Autograd: PyTorch package for differentiation of all operations on Tensors
# Variable are wrappers around Tensors that allow easy automatic differentiation
from torch.autograd import Variable

import time

import sys

# 'nn' Help us in creating & training of neural network
from torch import nn

# Contains definition for models for addressing different tasks i.e. image classification, object detection e.t.c.
from torchvision import models

from skimage import img_as_ubyte
import warnings
warnings.filterwarnings("ignore")


# Interaction with the OS
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# Used for DL applications, computer vision related processes
import torch
import torchvision

# For image preprocessing
from torchvision import transforms

# Combines dataset & sampler to provide iterable over the dataset
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

import numpy as np
import cv2

# To recognise face from extracted frames
import face_recognition

# Autograd: PyTorch package for differentiation of all operations on Tensors
# Variable are wrappers around Tensors that allow easy automatic differentiation
from torch.autograd import Variable

import time

import sys

# 'nn' Help us in creating & training of neural network
from torch import nn

# Contains definition for models for addressing different tasks i.e. image classification, object detection e.t.c.
from torchvision import models

from skimage import img_as_ubyte
import warnings
warnings.filterwarnings("ignore")

video_path = ""

detectOutput = []


# Creating Model Architecture

class Model(nn.Module):
  def __init__(self, num_classes, latent_dim= 2048, lstm_layers=1, hidden_dim=2048, bidirectional=False):
    super(Model, self).__init__()

    # returns a model pretrained on ImageNet dataset
    model = models.resnext50_32x4d(pretrained= True)

    # Sequential allows us to compose modules nn together
    self.model = nn.Sequential(*list(model.children())[:-2])

    # RNN to an input sequence
    self.lstm = nn.LSTM(latent_dim, hidden_dim, lstm_layers, bidirectional)

    # Activation function
    self.relu = nn.LeakyReLU()

    # Dropping out units (hidden & visible) from NN, to avoid overfitting
    self.dp = nn.Dropout(0.4)

    # A module that creates single layer feed forward network with n inputs and m outputs
    self.linear1 = nn.Linear(2048, num_classes)

    # Applies 2D average adaptive pooling over an input signal composed of several input planes
    self.avgpool = nn.AdaptiveAvgPool2d(1)



  def forward(self, x):
    batch_size, seq_length, c, h, w = x.shape

    # new view of array with same data
    x = x.view(batch_size*seq_length, c, h, w)

    fmap = self.model(x)
    x = self.avgpool(fmap)
    x = x.view(batch_size, seq_length, 2048)
    x_lstm,_ = self.lstm(x, None)
    return fmap, self.dp(self.linear1(x_lstm[:,-1,:]))




im_size = 112

# std is used in conjunction with mean to summarize continuous data
mean = [0.485, 0.456, 0.406]

# provides the measure of dispersion of image grey level intensities
std = [0.229, 0.224, 0.225]

# Often used as the last layer of a nn to produce the final output
sm = nn.Softmax()

# Normalising our dataset using mean and std
inv_normalize = transforms.Normalize(mean=-1*np.divide(mean, std), std=np.divide([1,1,1], std))

# For image manipulation
def im_convert(tensor):
  image = tensor.to("cpu").clone().detach()
  image = image.squeeze()
  image = inv_normalize(image)
  image = image.numpy()
  image = image.transpose(1,2,0)
  image = image.clip(0,1)
  cv2.imwrite('./2.png', image*255)
  return image

# For prediction of output  
def predict(model, img, path='./'):
  # use this command for gpu    
  # fmap, logits = model(img.to('cuda'))
  fmap, logits = model(img.to())
  params = list(model.parameters())
  weight_softmax = model.linear1.weight.detach().cpu().numpy()
  logits = sm(logits)
  _, prediction = torch.max(logits, 1)
  confidence = logits[:, int(prediction.item())].item()*100
  print('confidence of prediction: ', logits[:, int(prediction.item())].item()*100)
  return [int(prediction.item()), confidence]


# To validate the dataset
class validation_dataset(Dataset):
  def __init__(self, video_names, sequence_length = 60, transform=None):
    self.video_names = video_names
    self.transform = transform
    self.count = sequence_length

  # To get number of videos
  def __len__(self):
    return len(self.video_names)

  # To get number of frames
  def __getitem__(self, idx):
    video_path = self.video_names[idx]
    frames = []
    a = int(100 / self.count)
    first_frame = np.random.randint(0,a)
    for i, frame in enumerate(self.frame_extract(video_path)):
      faces = face_recognition.face_locations(frame)
      try:
        top,right,bottom,left = faces[0]
        frame = frame[top:bottom, left:right, :]
      except:
        pass
      frames.append(self.transform(frame))
      if(len(frames) == self.count):
        break
    frames = torch.stack(frames)
    frames = frames[:self.count]
    return frames.unsqueeze(0)

  # To extract number of frames
  def frame_extract(self, path):
    vidObj = cv2.VideoCapture(path)
    success = 1
    while success:
      success, image = vidObj.read()
      if success:
        yield image


def detectFakeVideo(videoPath):
    im_size = 112
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_transforms = transforms.Compose([
                                        transforms.ToPILImage(),
                                        transforms.Resize((im_size,im_size)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean,std)])
    path_to_videos= [videoPath]
    print(path_to_videos)
    video_dataset = validation_dataset(path_to_videos,sequence_length = 20,transform = train_transforms)
    # use this command for gpu
    # model = Model(2).cuda()
    model = Model(2)
    path_to_model = 'model/df_model.pt'
    model.load_state_dict(torch.load(path_to_model, map_location=torch.device('cpu')))
    model.eval()
    for i in range(0,len(path_to_videos)):
        print(path_to_videos[i])
        prediction = predict(model,video_dataset[i],'./')
        if prediction[0] == 1:
            print("REAL")
        else:
            print("FAKE")
    return prediction











from email.mime.base import MIMEBase
from email import encoders
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import pandas as pd
import sqlite3
import smtplib

def mail(data):
    mail_content = {} 
    mail_content['subject'] = 'Fake Video'
    mail_content['text'] = f"Dear Advocate,\n\nThis mail to bring your attention to a concerning matter regarding a fake video that has surfaced online.\n\nComplaint from {data['name']}\n{data['complaint']},\nName: {data['name']}\nPhone Num: {data['pnum']}\nEmail ID: {data['email']}\n\nThank you for your attention to this urgent matter.\nBest regards"

    mail_content['attachment_path'] = "detection_app/static/"+data['path']        
    send_mail(data['email'],mail_content)

def send_mail(email,mail_content):
    msg = MIMEMultipart()
    # msg = MIMEText(f"Dear {name}\n\n Thank you very much for attending the technical interview process.Iam delighted to inform that you are qualified in the technical interview and has been selected for HR interview.\n\nCONGRATULATIONS!\n\nFurther details for HR round will be shared later")
    msg['Subject'] = mail_content['subject']
    msg['From'] = 'armchatbot3@gmail.com'
    msg['To'] = email
    msg.attach(MIMEText(mail_content['text']))
    # msg.attach(MIMEText(f"Dear {name}\n\n Thank you very much for attending the technical interview process.Iam delighted to inform that you are qualified in the technical interview and has been selected for HR interview.\n\nCONGRATULATIONS!\n\nFurther details for HR round will be shared later"))

    if mail_content['attachment_path']:
        part = MIMEBase('application', "octet-stream")
        part.set_payload(open(mail_content['attachment_path'], "rb").read())
        encoders.encode_base64(part)
        part.add_header('Content-Disposition', 'attachment; filename="video.mp4"')
        msg.attach(part)

        # Send the message using SMTP
    smtp_server = 'smtp.gmail.com'
    smtp_port = 587
    smtp_username = 'armchatbot3@gmail.com'
    smtp_password = 'vcefmskrqgtcuvjq'
    smtp_conn = smtplib.SMTP(smtp_server, smtp_port)
    smtp_conn.starttls()
    smtp_conn.login(smtp_username, smtp_password)
    smtp_conn.sendmail(smtp_username, [msg['To']], msg.as_string())
    smtp_conn.quit()

