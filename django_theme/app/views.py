import PIL
from django.http import JsonResponse
from django.shortcuts import render,redirect
from django.core.files.storage import FileSystemStorage
from django.utils.baseconv import base64
from keras.preprocessing.image import ImageDataGenerator,load_img, img_to_array
from django.views.generic import TemplateView
from django.http import HttpResponse
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
import pickle
import re
import io
import tensorflow as tf
import os

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.layers import Input, Dense,Conv2D, BatchNormalization, Activation, Flatten
from tensorflow.keras.models import Model
from tensorflow import keras
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split


from io import  StringIO
from matplotlib import pylab
from pylab import *

from app.tasks import process, process_training
from tensorflow.keras import backend as K
from app.forms import JobForm
from celery.result import AsyncResult
from app.models import Tasks
from django.views.decorators.http import require_GET
from django.views.decorators.http import require_http_methods

# Create your views here.

MODEL=load_model('.\\models\\covidnet.hdf5')
ChestCR_model = load_model(r'C:\Users\4472829\PycharmProjects\Jupyter_notebook\covidCRnet.hdf5')
covid_kaggle_model = load_model(r'C:\Users\4472829\PycharmProjects\Jupyter_notebook\covid_kagglenet.hdf5')
untrain_model = load_model('.\\models\\untrained_model.hdf5')
data = 'C:\\Users\\4472829\\Downloads\\covid19\\dataset'
normal_scan_path = r"M:\dept\Dept_MachineLearning\Staff\ML Engineer\Naveena Gorre\Datasets\Covid_MIDRC\Covid_Classification\Covid_negative"
abnormal_scan_path = r"M:\dept\Dept_MachineLearning\Staff\ML Engineer\Naveena Gorre\Datasets\Covid_MIDRC\Covid_Classification\Covid_positive"


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
                return redirect('users')
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
    # def table(request):
    #     return render(request, 'tables.html')
    # testing comment

    def loadData(request):
        if request.method == 'POST':
            return render(request,'ui-database.html')
        return render(request,'ui-database.html')

    @require_http_methods(["GET", "POST"])
    def run(request):
        form = JobForm(request.POST)
        if request.method == "POST":
            if form.is_valid():
                data = form.cleaned_data
                job_name = data['job_name']
                process.delay(job_name=job_name)
                return render(request, 'run.html',context={'form': JobForm,
                                           'message': f'{job_name} dispatched...'})
            else:
                print('form is invalid')
        else:
            return render(request, 'run.html',
                              context={'form': JobForm})
        #return render(request, 'run.html')

    def run_training(request):
       print("request is:", request)
       if request.method == 'POST':
            Epochs = int(request.POST.get('epochVal'))
            Augument = request.POST.getlist('augment')
            print("augumentation value is:  ", Augument)
            Batchsize = int(request.POST.get('batchsizeVal'))
            LearningRate = request.POST.get('learnrateVal')
            loss = request.POST.get('lossVal')
            optimizer = request.POST.get('optimizerVal')
            model_input = request.POST.get('vggVal')
            aug_value = request.POST.get('augVal')
            if model_input == "Fine tuning":
                process_training.delay(Epochs,LearningRate,Batchsize,job_name = model_input)
                #context = ChestCR_model.history['accuracy']
                return render(request, 'training.html', context={'message': f'You have chosen {model_input}'})

            elif model_input == "Training from scratch":
                process_training.delay(job_name=model_input)
                return render(request, 'training.html', context={'message': f'You have chosen {model_input}'})
       return render(request, "training.html")

    @require_GET
    def monitor_training(request):
        info = Home.track_jobs()
        return render(request, 'monitor.html', context={'info': info})

    @staticmethod
    def track_jobs():
        entries = Tasks.objects.all()
        information = []
        for item in entries:
            progress = 100  # max value for bootstrap progress
            # bar, when  the job is finished
            result = AsyncResult(item.task_id)
            if isinstance(result.info, dict):
                progress = result.info['progress']
            information.append([item.job_name, result.state,
                                    progress, item.task_id])
        return information

    @require_GET
    def monitor(request):
        info = Home.track_jobs()
        return render(request, 'monitor.html', context={'info': info})

    @require_GET
    def cancel_job(request, task_id=None):
        result = AsyncResult(task_id)
        result.revoke(terminate=True)
        info = Home.track_jobs()
        return render(request, 'monitor.html', context={'info': info})

    @require_GET
    def delete_job(request, task_id=None):
        a = Tasks.objects.filter(task_id=task_id)
        a.delete()
        info = Home.track_jobs()
        return render(request, 'monitor.html', context={'info': info})


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


    def conv2d_block(input_tensor, n_filters, kernel_size=3):
        x = Conv2D(filters=n_filters, kernel_size=kernel_size, kernel_initializer="he_normal", padding="same")(input_tensor)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Conv2D(filters=n_filters, kernel_size=kernel_size, kernel_initializer="he_normal", padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Conv2D(filters=2 * n_filters, strides=(2, 2), kernel_size=kernel_size, kernel_initializer="he_normal",
                   padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        return x


    def model4(n_classes, input_shape):
        '''
           Classifier following encoder with random initialization (inspired by VGG structure)
           input size must be fixed due to Flat+Dense

           n_classes: number of ground truth classes
           input_shape: shape of single input datum
           '''
        global Input
        Input = Input(input_shape, K.learning_phase())
        x = Home.conv2d_block(Input, 8)
        x = Home.conv2d_block(Input, 16)
        x = Home.conv2d_block(Input, 32)
        x = Home.conv2d_block(Input, 64)
        flat = Flatten()(x)
        proj = Dense(1024, activation="relu")(flat)
        soft = Dense(n_classes, activation="softmax")(proj)

        model = Model(inputs=[Input], outputs=[soft])

        return model

    def training_model(request):
        if request.method == 'POST':
            Epochs = int(request.POST.get('epochVal'))
            batchsize = int(request.POST.get('batchsizeVal'))
            learningrate = request.POST.get('learnrateVal')
            loss = request.POST.get('lossVal')
            optimizer = request.POST.get('optimizerVal')
            model_input = request.POST.get('vggVal')
            if model_input == "Fine tuning":
                # load data paths, process scans to obtain pixel array and generate labels
                print('You have chosen:',model_input,",extracting the pixel array of dicom images to train the model...")
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
                    epochs=5,
                    batch_size=10,
                    validation_data=(X_val, y_val),
                )

            context = ChestCR_model.history['accuracy']
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
                print(filePathName)
                test_image = '.'+filePathName
                #img = tf.keras.preprocessing.image.load_img(test_image,target_size=(Image_Height, Image_Width))
                #img_nparray = tf.keras.preprocessing.image.img_to_array(img)
                #input_Batch = np.array([img_nparray])
                #prediction = MODEL.predict(input_Batch)
                test_scan = Home.process_scan(test_image)
                test_scan_pred = np.expand_dims(test_scan, axis=-1)
                #print(ChestCR_model.summary())
                prediction = ChestCR_model.predict(np.expand_dims(test_scan_pred, axis=0))[0]
                scores = [1 - prediction[0], prediction[0]]
                class_names = ["Covid -ve", "Covid +ve"]
                for score, name in zip(scores, class_names):
                    print("This model is %.2f percent confident that Covid scan is %s"
                          % ((100 * score), name))
                    final_score1 = 100*scores[0]
                    final_name1 = class_names[0]
                    final_score2=100*scores[1]
                    final_name2 = class_names[1]
                #final_pred = np.argmax(prediction,axis=1)
                #for pred in final_pred:
                    #if pred == 1:
                       # label ='normal'
                    #elif pred == 0:
                        #label='covid'
                #context = {'message': f'You have chosen {model_input}'}
                context={'message': f'Model prediction: %.2f' % ((final_score1)),
                         'message1': f'Label: %s' % (final_name1),}
                return render(request,'testing.html',context)
        return render(request,'testing.html')

    def plot_acc(request):
        covidCR_model_2 = pickle.load(open(r'C:\Users\4472829\PycharmProjects\Jupyter_notebook\finetuning_imagenet_hpt', "rb"))
        fig, ax = plt.subplots(1, 2, figsize=(14, 5))
        ax[0].plot(covidCR_model_2['acc'])
        ax[0].plot(covidCR_model_2['val_acc'])
        ax[0].set_title('model accuracy')
        ax[0].set_ylabel('accuracy')
        ax[0].set_xlabel('epoch')
        ax[0].legend(['train', 'test'], loc='upper left')
        ax[1].plot(covidCR_model_2['loss'])
        ax[1].plot(covidCR_model_2['val_loss'])
        ax[1].set_title('model loss')
        ax[1].set_ylabel('loss')
        ax[1].set_xlabel('epoch')
        ax[1].legend(['train', 'test'], loc='upper left')
        canvas = FigureCanvasTkAgg(fig)
        response = HttpResponse(content_type='image/png')
        canvas.print_png(response)
        return response

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
        DNN_layers = [layer.name for layer in covid_kaggle_model.layers]
        if request.method == 'POST':
            layer = request.POST.get('layers')
            fileObj = request.FILES['filePath']
            fs = FileSystemStorage()
            filePathName = fs.save(fileObj.name, fileObj)
            filePathName = fs.url(filePathName)
            test_image = '.' + filePathName
            #input_test = Home.process_scan(test_image)
            #activations = get_activations(ChestCR_model, np.expand_dims(input_test, axis=0), layer)
            input_test = Home.preprocess_image(img_path=test_image, model=covid_kaggle_model,
                                               resize=(Image_Height, Image_Width))
            activations = get_activations(covid_kaggle_model, input_test, layer)
            heatMapImgPath = display_heatmaps(activations, input_test, directory=r'./media/', save=True)
            print( heatMapImgPath)
            return render(request, 'heat_maps.html',
                      {"layer_name": layer, "DNN_layers": DNN_layers, 'filePathName': filePathName,
                       'HeatMapImgPath':  heatMapImgPath})
        return render(request, 'heat_maps.html', {"DNN_layers": DNN_layers})
    
    def activation_maps(request):
        DNN_layers = [layer.name for layer in covid_kaggle_model.layers]
        if request.method == 'POST':
            layer = request.POST.get('layers')
            fileObj=request.FILES['filePath']
            fs=FileSystemStorage()
            filePathName=fs.save(fileObj.name,fileObj)
            filePathName=fs.url(filePathName)
            test_image = '.'+filePathName
            #input_test = Home.process_scan(test_image)
            #activations = get_activations(ChestCR_model, np.expand_dims(input_test, axis=0), layer)
            input_test = Home.preprocess_image(img_path=test_image,model=covid_kaggle_model,resize=(Image_Height, Image_Width))
            activations = get_activations(covid_kaggle_model, input_test,layer)
            activationMapImgPath = display_activations(activations, directory=r'./media/', save=True)
            print(activationMapImgPath)
            return render(request,'activation_maps.html',
                          {"layer_name": layer,"DNN_layers": DNN_layers,'filePathName':filePathName,
                                                          'ActivationMapImgPath':activationMapImgPath})
        return render(request,'activation_maps.html',{"DNN_layers": DNN_layers})

    def shapley_values(request):
        if request.method == 'POST':
            print(request)
        return render(request, 'shapley_value.html')

