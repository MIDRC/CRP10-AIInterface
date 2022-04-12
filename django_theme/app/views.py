from django.shortcuts import render,redirect
from django.core.files.storage import FileSystemStorage
from keras.preprocessing.image import ImageDataGenerator,load_img, img_to_array
from django.views.generic import TemplateView
from django.contrib.auth.models import User
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.decorators import login_required
from django.contrib.auth.mixins import LoginRequiredMixin
import shap
from PIL import Image
from keras.preprocessing import image
from keras.models import load_model
from keract import get_activations
from keract import display_activations,display_heatmaps
import numpy as np
import cv2
import pydicom
import re
import tensorflow as tf
import os
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.layers import Input, Dense,Conv2D, BatchNormalization, Activation, Flatten
from tensorflow.keras.models import Model
from tensorflow import keras
from sklearn.model_selection import train_test_split
# Create your views here.

MODEL=load_model('.\\models\\covidnet.hdf5')
untrain_model = load_model('.\\models\\untrained_model.hdf5')
data = 'C:\\Users\\4472829\\Downloads\\covid19\\dataset'


Image_Height, Image_Width = 150,150 
BATCH_SIZE, EPOCHS = 5,5

class Registration(TemplateView):
    template_name = 'base.html'

    def users(request):
        count = User.objects.count()
        return render(request, 'users.html', {
            'count': count
        })

    def signup(request):
        if request.method == 'POST':
            form = UserCreationForm(request.POST)
            if form.is_valid():
                form.save()
                return redirect('index')
        else:
            form = UserCreationForm()
        return render(request, 'registration/signup.html', {
            'form': form
        })

    @login_required
    def secret_page(request):
        return render(request, 'secret_page.html')
    # def index(request):
    #     #context = {'a': 1}
    #     return render(request, 'index.html')

    # class SecretPage(LoginRequiredMixin, TemplateView):
    #     template_name = 'secret_page.html'


class Home(TemplateView):
   # template_name = 'index.html'

    def login_base(request):
        return render(request, 'base_auth.html')

    @login_required
    def index(request):
         context = {'a': 1}
         return render(request, 'index.html')

    # testing comment
    def loadData(request):
        if request.method == 'POST':
            fileObj=request.FILES['filePath']
            fs=FileSystemStorage()
            filePathName=fs.save(fileObj.name,fileObj)
            print(filePathName)
            filePathName=fs.url(filePathName)
            context={'filePathName':filePathName}
            return render(request,'Multimodality.html',context)
        return render(request,'Multimodality.html')

    def process_scan(filepath):
        scan = pydicom.read_file(str(filepath))
        scan = scan.pixel_array
        scan = cv2.resize(scan, (224, 224))
        return scan


    def scanpath(Base_path):
       scans = []
       for root, dirs, files in os.walk(Base_path):
           for fname in files:
               scans.append(os.path.join(root, fname))
       return scans
    def pixelarray (normal_scan_path,abnormal_scan_path):
        normal_scan_paths = Home.scanpath(normal_scan_path)
        abnormal_scan_paths = Home.scanpath(abnormal_scan_path)
        #
        normal_scans = np.array([Home.process_scan(path) for path in normal_scan_paths])
        abnormal_scans = np.array([Home.process_scan(path) for path in abnormal_scan_paths])
        #
        normal_labels = np.array([0 for _ in range(len(normal_scans))])
        abnormal_labels = np.array([1 for _ in range(len(abnormal_scans))])
        # Perform data split for training, validation, testing
        X_train, X_test, y_train, y_test = train_test_split(np.concatenate((abnormal_scans, normal_scans)),
                                                            np.concatenate((abnormal_labels, normal_labels)),
                                                            test_size=0.2, shuffle=True, random_state=8)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=8)
        return X_train, y_train, X_test, y_test, X_val, y_val


    def model2(n_classes=2, input_shape=(224, 224, 1)):
       vgg_model = VGG19(include_top=False, weights=None, input_shape=input_shape)
       flat = Flatten()(vgg_model.layers[-1].output)
       proj = Dense(1024, activation='relu')(flat)
       soft = Dense(1, activation='sigmoid')(proj)
       model = Model(inputs=vgg_model.inputs, outputs=soft)
       return model


    def training_model(request):
        if request.method == 'POST':
            Epochs = int(request.POST.get('epochVal'))
            batchsize = int(request.POST.get('batchsizeVal'))
            learningrate = request.POST.get('learnrateVal')
            loss = request.POST.get('lossVal')
            optimizer = request.POST.get('optimizerVal')
            model_input = request.POST.get('vggVal')
            print(model_input)
            if model_input == "Model2":
                # load data paths, process scans to obtain pixel array and generate labels
                print('You have chosen:',model_input,",extracting the pixel array of dicom images to train the model...")
                normal_scan_path = r"M:\dept\Dept_MachineLearning\Staff\ML Engineer\Naveena Gorre\Datasets\Covid_MIDRC\Covid_Classification\Covid_negative"
                abnormal_scan_path = r"M:\dept\Dept_MachineLearning\Staff\ML Engineer\Naveena Gorre\Datasets\Covid_MIDRC\Covid_Classification\Covid_positive"
                X_train, y_train, X_test, y_test, X_val, y_val = Home.pixelarray(normal_scan_path,abnormal_scan_path)
                CRcl_model = Home.model2()
                CRcl_model.compile(
                    loss="binary_crossentropy",
                    optimizer=keras.optimizers.Adam(learning_rate=0.001),
                    metrics=["acc"],
                )
                CRcl_model.fit(
                    X_train,
                    y_train,
                    epochs=25,
                    batch_size=10,
                    validation_data=(X_val, y_val),
                )

           #  train_datagen = ImageDataGenerator(rescale=1./255,
           #  shear_range=0.2,
           #  zoom_range=0.2,
           #  horizontal_flip=True,
           #  validation_split=0.25)
           #
           #  train_generator = train_datagen.flow_from_directory(
           #  data,
           #  target_size=(Image_Height, Image_Width),
           #  batch_size=batchsize,
           #  class_mode='binary',
           #  subset='training')
           #
           #  validation_generator = train_datagen.flow_from_directory(
           #  data,
           #  target_size=(Image_Height, Image_Width),
           #  batch_size=batchsize,
           #  class_mode='binary',
           #  shuffle= False,
           #  subset='validation')
           # # untrain_model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
           #  covid_model = untrain_model.fit(
           #  train_generator,
           #  steps_per_epoch = train_generator.samples // batchsize,
           #  validation_data = validation_generator,
           #  validation_steps = validation_generator.samples // batchsize,
           #  epochs = Epochs)
            
            #context = covid_model.history['accuracy']
            return render(request,'training.html',{"context": context,'epochs':Epochs})
        return render(request,"training.html")
    
    def removespaces(file_with_space):
        return os.rename(file_with_space,file_with_space.replace(' ', '_'))

    def testing(request):
        if request.method == 'POST':
            fileObj=request.FILES['filePath']
            fs=FileSystemStorage()
            fileObj_search = bool(re.search(r"\s", str(fileObj)))
            if fileObj_search == True:
                #test_res = Home.removespaces(fileObj)
                return render(request,'error.html')
            else:   
                filePathName=fs.save(fileObj.name,fileObj)
                filePathName=fs.url(filePathName)  
                test_image = '.'+filePathName
                img = tf.keras.preprocessing.image.load_img(test_image,target_size=(Image_Height, Image_Width))
                img_nparray = tf.keras.preprocessing.image.img_to_array(img)
                input_Batch = np.array([img_nparray])   
                prediction = MODEL.predict(input_Batch)
                print(prediction)
                final_pred = np.argmax(prediction,axis=1)
                for pred in final_pred:
                    if pred == 1:
                        label ='normal'
                    elif pred == 0:
                        label='covid'
                context={'filePathName':filePathName,'predictedLabel':label}
                return render(request,'testing.html',context)
        return render(request,'testing.html')
    
    def preprocess_image(img_path, model=None, rescale=255, resize=(256, 256)):    
        assert type(img_path) == str, "Image path must be a string"
        assert (
            type(rescale) == int or type(rescale) == float
        ), "Rescale factor must be either a float or int"
        assert (
            type(resize) == tuple and len(resize) == 2
        ), "Resize target must be a tuple with two elements"
    
        img = load_img(img_path)
        img = img_to_array(img)
        img = img / float(rescale)
        img = cv2.resize(img, resize)
        if model != None:
            if len(model.input_shape) == 4:
                img = np.expand_dims(img, axis=0)
        return img
            
    def heat_maps(request):
        DNN_layers = [layer.name for layer in untrain_model.layers]
        if request.method == 'POST':
            layer = request.POST.get('layers')
            fileObj=request.FILES['filePath']
            fs=FileSystemStorage()
            filePathName=fs.save(fileObj.name,fileObj)
            filePathName=fs.url(filePathName)
            test_image = '.'+filePathName
            input_test = Home.preprocess_image(img_path=test_image,model=MODEL,resize=(Image_Height, Image_Width))
            activations = get_activations(MODEL, input_test,layer)
            heatMapImgPath = display_heatmaps(activations, input_test, directory=r'./media/', save=True)
            return render(request,'heat_maps.html',{"layer_name": layer,"DNN_layers": DNN_layers,'filePathName':filePathName,'heatMapImgPath':heatMapImgPath})
        return render(request,'heat_maps.html',{"DNN_layers": DNN_layers})
    
    def activation_maps(request):
        DNN_layers = [layer.name for layer in untrain_model.layers]
        if request.method == 'POST':
            layer = request.POST.get('layers')
            fileObj=request.FILES['filePath']
            fs=FileSystemStorage()
            filePathName=fs.save(fileObj.name,fileObj)
            filePathName=fs.url(filePathName)
            test_image = '.'+filePathName
            input_test = Home.preprocess_image(img_path=test_image,model=MODEL,resize=(Image_Height, Image_Width))
            activations = get_activations(MODEL, input_test,layer)
            activationMapImgPath = display_activations(activations, directory=r'./media/', save=True)
            print(activationMapImgPath)
            return render(request,'activation_maps.html',{"layer_name": layer,"DNN_layers": DNN_layers,'filePathName':filePathName,'ActivationMapImgPath':activationMapImgPath})
        return render(request,'activation_maps.html',{"DNN_layers": DNN_layers})

    def shapley_values(request):
        if request.method == 'POST':
            print(request)
        return render(request, 'shapley_value.html')

