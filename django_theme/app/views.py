import PIL, shap, cv2, pydicom, pickle, re, io, os, copy, math
from django.shortcuts import render, redirect, HttpResponse
import subprocess
import time
from django.core.files.storage import FileSystemStorage
from keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.utils import load_img, img_to_array
from django.views.generic import TemplateView
from django.http import HttpResponse
from django.contrib.auth.models import User
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.decorators import login_required
from keras.models import load_model
from keract import get_activations
from keract import display_activations, display_heatmaps
import numpy as np
import tensorflow as tf

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.layers import Input, Dense, Conv2D, BatchNormalization, Activation, Flatten
from tensorflow.keras.models import Model
from tensorflow import keras
from sklearn.model_selection import train_test_split
from pylab import *

from app.tasks import process, process_training
from tensorflow.keras import backend as K
from app.forms import JobForm
from celery.result import AsyncResult
from app.models import Tasks
from django.views.decorators.http import require_GET
from django.views.decorators.http import require_http_methods
from django.conf import settings
import sys

# added for testing/visualization
import matplotlib.pyplot as plt
from django.shortcuts import render
from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img, ImageDataGenerator

# added for augmentation
from scipy.ndimage import rotate
import random
import pandas as pd
import shutil
from sklearn.model_selection import train_test_split
import SimpleITK as sitk
from glob import glob
import pydicom as dicom

# for viewing
# maybe import media folder?
from PIL import Image
from tensorflow.keras.applications.inception_v3 import preprocess_input, decode_predictions
# wrapper
from scripts.wrappers import *
# from scripts.displays import *
from scripts.data_visualization import *
import urllib
from app import tasks

MODEL = load_model(settings.CHESTCR_MODEL)
if settings.USING_MODELS:  # local settings
    ChestCR_model = load_model(settings.CHESTCR_MODEL)
    covidCR_model_2 = settings.COVIDCR_MODEL_2
    covid_kaggle_model = settings.COVID_KAGGLE_MODEL
    untrain_model = load_model(settings.UNTRAIN_MODEL)
    data = settings.DATA
    normal_scan_path = settings.NORMAL_SCAN_PATH
    abnormal_scan_path = settings.ABNORMAL_SCAN_PATH
else:

    ChestCR_model = 'None'
    covidCR_model_2 = 'None'
    covid_kaggle_model = 'None'
    untrain_model = 'None'
    data = 'None'
    normal_scan_path = 'None'
    abnormal_scan_path = 'None'

Image_Height, Image_Width = 150, 150
BATCH_SIZE, EPOCHS = 5, 5


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

    def loadData(request):
        if request.method == 'POST':
            return render(request, 'ui-database.html')
        return render(request, 'ui-database.html')

    @require_http_methods(["GET", "POST"])
    def run(request):
        form = JobForm(request.POST)
        if request.method == "POST":
            if form.is_valid():
                data = form.cleaned_data
                job_name = data['job_name']  # put time/date here later for better identification
                process.delay(job_name=job_name)
                return render(request, 'run.html', context={'form': JobForm,
                                                            'message': f'{job_name} dispatched...'})
            else:
                print('form is invalid')
        else:
            return render(request, 'run.html',
                          context={'form': JobForm})
        # return render(request, 'run.html')

    def run_training(request):
        print("request is:", request)
        if request.method == 'POST':
            Epochs = int(request.POST.get('epochVal'))
            Batchsize = int(request.POST.get('batchsizeVal'))
            LearningRate = float(request.POST.get('learnrateVal'))
            loss = request.POST.get('lossVal')

            if (request.POST.get('augVal') == 'No'):
                optimizer = request.POST.get('optimizerVal')
                model_input = request.POST.get('vggVal')
                if model_input == "Fine tuning":
                    process_training.delay(Epochs, LearningRate, Batchsize, loss, optimizer, job_name=model_input)
                    return render(request, 'training.html', context={
                        'message': f'To see the status of the training, click here or go to "Load Database -> Monitor Jobs"'})

                elif model_input == "Transfer learning":
                    process_training.delay(Epochs, LearningRate, Batchsize, loss, optimizer,job_name=model_input)  # change later
                    return render(request, 'training.html', context={
                        'message': f'To see the status of the training, click here, click the gear, or go to "Load Database -> Monitor Jobs"'})

            elif (request.POST.get('augVal') == 'Yes'):
                optimizer = request.POST.get('optimizerVal')
                model_input = request.POST.get('vggVal')
                if model_input == "Fine tuning":
                    process_training.delay(Epochs, LearningRate, Batchsize, loss, optimizer,job_name=model_input)
                    return render(request, 'training.html', context={
                        'message': f'To see the status of the training, click here or go to "Load Database -> Monitor Jobs"'})

                elif model_input == "Transfer learning":
                    # process_training.delay(job_name=model_input) #commented out for now due to errors
                    process_training.delay(Epochs, LearningRate, Batchsize, loss, optimizer,
                                           job_name=model_input)  # change later
                    return render(request, 'training.html', context={
                        'message': f'To see the status of the training, click here, click the gear, or go to "Load Database -> Monitor Jobs"'})
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
            if result.info is not None and isinstance(result.info, dict):
                if 'progress' in result.info:
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

    # augmentations done here, maybe make augs seperate functions if other processes need it
    def process_scan(Augment, filepath):
        scan = pydicom.read_file(str(filepath))
        filepath = filepath.replace('/', '\\')  # stablizes function for whatever filepath could be requested as
        augName = filepath[(filepath.rindex('\\')) + 1:]
        # if "training" or "TRAINING" in augName:
        #     print("yerr")
        augName = augName.replace("testing", "Testing")
        augName = augName.replace("TESTING", "Testing")
        scan = scan.pixel_array
        scan = cv2.resize(scan, (224, 224))  # size is set to 224 by 224, may want to not hard code it in the future

        # for easy showing of aug vs og, change so that "debug/condition will run this part only"
        if os.path.exists(settings.OGIMG): shutil.rmtree(settings.OGIMG, ignore_errors=True)
        os.mkdir(settings.OGIMG)
        temp = sitk.GetImageFromArray(scan)
        tempName = settings.OGIMG + settings.SEP + augName
        # print(tempName)
        sitk.WriteImage(temp, tempName)
        file = pydicom.dcmread(tempName)
        dcm_jpg(file, tempName)
        # print(Augment)
        # rot = any('Rotate' in sub for sub in Augment)

        for i in Augment:
            # reason why the if statements are staggered are for scalability/stability
            # bad inputs (if they somehow get to this point) will not crash the program
            # staggers prevent accessing anything out of index and all that
            # also redundant checks make sure the performed augment is the correct one,
            # as any future augmentations with similar lengths/names will not be called improperly

            # print(i)
            # time.sleep(5) # temporary, just allows for easier testing
            # print('space')

            if (len(i) > 0):
                if (i[0] == 'R' and len(i) > 6):
                    if (i[0:7] == 'Rotate-'):
                        temp = len(i) - 7  # if temp is zero, then it is random (imp later)
                        if (temp > 0):
                            # check if none/random was selected later too
                            scan = rotate(scan, int(i[7:]))
                            scan = cv2.resize(scan, (224, 224))  # must resize after rotation

                elif (i[0] == 'Z' and len(i) > 5):
                    if (i[0:5] == 'Zoom-'):
                        temp = int(int(i[5:]) / 100)
                        print(temp)
                        np.kron(scan, np.ones((temp, temp)))  # zoom in by whatever factor

                elif (i[0] == 'V' and len(i) == 5):
                    if (i == 'VFlip'): scan = np.flip(scan, 1)

                elif (i[0] == 'H' and len(i) == 5):
                    if (i == 'HFlip'): scan = np.flip(scan, 0)

                elif (i[0] == 'S' and len(i) == 5):
                    if (i == 'Shear'):
                        for x in range(scan.shape[1]):
                            scan[:, x] = np.roll(scan[:, 1], x)
                            scan = cv2.resize(scan, (224, 224))

        # # old code, will remove once everything is stable

        # if ('Rotate' in Augment):
        #     scan = rotate(scan, 45*random.randint(1,2))
        #     scan = cv2.resize(scan, (224,224)) #must resize after rotation

        # if ('Zoom' in Augment):
        #     np.kron(scan, np.ones((2,2))) #zoom in by factor of 2

        # if ('VFlip' in Augment):
        #     scan = np.flip(scan, 1)

        # if ('HFlip' in Augment):
        #     scan = np.flip(scan,0)

        # if ('Shear' in Augment):
        #     for i in range (scan.shape[1]):
        #         scan[:,i] = np.roll(scan[:,1],i)
        #         scan = cv2.resize(scan, (224,224))

        # if ('activation' in Augment):
        #     #aug = sitk.GetImageFromArray(scan)
        #     # augName = settings.ACTIVATION_MAPS + '\\' + augName
        #     # sitk.WriteImage(aug, augName)
        #     #file = pydicom.dcmread(settings.ACTIVATION_MAPS + "\\test.dcm")
        #     #dcm_jpg(file,  settings.ACTIVATION_MAPS + "\\test.dcm")
        #     #scan = preprocess_input(scan)
        #     aug = scan
        #     #solo(aug)

        if ('Rotate' in Augment):
            # scan = rotate(scan, random.uniform (0.0, 360.0)) #for any degree of rotation, 360 can be substituted with whatever upper bound
            scan = rotate(scan, 45 * random.randint(1, 2))
            scan = cv2.resize(scan, (224, 224))  # must resize after rotation

        if ('Zoom' in Augment):
            np.kron(scan, np.ones((2, 2)))  # zoom in by factor of 2

        if ('VFlip' in Augment):
            scan = np.flip(scan, 1)

        if ('HFlip' in Augment):
            scan = np.flip(scan, 0)

        if ('shear' in Augment):
            for i in range(scan.shape[1]):
                scan[:, i] = np.roll(scan[:, 1], i)
                scan = cv2.resize(scan, (224, 224))

        if ('activation' in Augment):
            # aug = sitk.GetImageFromArray(scan)
            # augName = settings.ACTIVATION_MAPS + '\\' + augName
            # sitk.WriteImage(aug, augName)
            # file = pydicom.dcmread(settings.ACTIVATION_MAPS + "\\test.dcm")
            # dcm_jpg(file,  settings.ACTIVATION_MAPS + "\\test.dcm")
            # scan = preprocess_input(scan)
            aug = scan
            # solo(aug)
        # below is for testing, saves aug for viewing
        # if you want to view as dicom file, comment out dcm_jpg (may add function to choose later)
        if os.path.exists(settings.AUGIMG): shutil.rmtree(settings.AUGIMG, ignore_errors=True)
        os.mkdir(settings.AUGIMG)
        aug = sitk.GetImageFromArray(scan)
        augName = settings.AUGIMG + settings.SEP + augName
        sitk.WriteImage(aug, augName)
        file = pydicom.dcmread(augName)
        dcm_jpg(file, augName)

        return scan

    def scanpath(Base_path):
        scans = []
        for root, dirs, files in os.walk(Base_path):
            for fname in files:
                scans.append(os.path.join(root, fname))
        return scans

    def pixelarray(Augment, normal_scan_path, abnormal_scan_path):
        normal_scan_paths = Home.scanpath(normal_scan_path)
        abnormal_scan_paths = Home.scanpath(abnormal_scan_path)

        # clears folder holding augmented images
        if os.path.exists(settings.AUGIMG):
            shutil.rmtree(settings.AUGIMG, ignore_errors=True)
        os.mkdir(settings.AUGIMG)
        if os.path.exists(settings.OGIMG): shutil.rmtree(settings.OGIMG, ignore_errors=True)
        os.mkdir(settings.OGIMG)
        normal_scans = np.array([Home.process_scan(Augment, path) for path in normal_scan_paths])
        abnormal_scans = np.array([Home.process_scan(Augment, path) for path in abnormal_scan_paths])

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
        x = Conv2D(filters=n_filters, kernel_size=kernel_size, kernel_initializer="he_normal", padding="same")(
            input_tensor)
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
        # add a layer of rotation here
        print("training model!")
        if request.method == 'POST':
            Epochs = int(request.POST.get('epochVal'))
            batchsize = int(request.POST.get('batchsizeVal'))
            learningrate = float(request.POST.get('learnrateVal'))
            loss = request.POST.get('lossVal')
            optimizer = request.POST.get('optimizerVal')
            model_input = request.POST.get('vggVal')
            if model_input == "Fine tuning":
                # load data paths, process scans to obtain pixel array and generate labels
                print('You have chosen:', model_input,
                      ",extracting the pixel array of dicom images to train the model...")
                X_train, y_train, X_test, y_test, X_val, y_val = Home.pixelarray(normal_scan_path, abnormal_scan_path)

                # HERE IS WHERE TRANSFORMATION WILL HAPPEN
                # if rotation to augment x trainbb

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
            return render(request, 'training.html', {"context": context, 'epochs': Epochs})
        return render(request, "training.html")

    def removespaces(file_with_space):
        return os.rename(file_with_space, file_with_space.replace(' ', '_'))

    def testing(request):
        if request.method == 'POST':
            fileObj = request.FILES['filePath']
            fs = FileSystemStorage()
            fileObj_search = bool(re.search(r"\s", str(fileObj)))
            if fileObj_search == True:
                # test_res = Home.removespaces(fileObj)
                return render(request, 'error.html')
            else:
                filePathName = fs.save(fileObj.name, fileObj)
                filePathName = fs.url(filePathName)
                print(filePathName)
                test_image = '.' + filePathName
                # img = tf.keras.preprocessing.image.load_img(test_image,target_size=(Image_Height, Image_Width))
                # img_nparray = tf.keras.preprocessing.image.img_to_array(img)
                # input_Batch = np.array([img_nparray])
                # prediction = MODEL.predict(input_Batch)
                test_scan = Home.process_scan("", test_image)
                test_scan_pred = np.expand_dims(test_scan, axis=-1)
                print(test_scan_pred)
                print(ChestCR_model.summary())
                prediction = ChestCR_model.predict(np.expand_dims(test_scan_pred, axis=0))[0]
                scores = [1 - prediction[0], prediction[0]]
                class_names = ["Covid -ve", "Covid +ve"]
                for score, name in zip(scores, class_names):
                    print("This model is %.2f percent confident that Covid scan is %s"
                          % ((100 * score), name))
                    final_score1 = 100 * scores[0]
                    final_name1 = class_names[0]
                    final_score2 = 100 * scores[1]
                    final_name2 = class_names[1]
                # final_pred = np.argmax(prediction,axis=0)
                # for pred in final_pred:
                #   if pred == 1:
                #      label ='normal'
                # elif pred == 0:
                #    label='covid'

                # context = {'message': f'You have chosen {model_input}'}

                # for testing purposes only, def change later
                # may be good to later add default as blank/error
                # this may also be returning badly
                sanitized = filePathName[(filePathName.rindex('/')):-4]
                sanitized = sanitized.replace("testing", "Testing")
                sanitized = sanitized.replace("TESTING", "Testing")
                ogImg = settings.OGIMG + sanitized + '.jpg'
                # ogImg = settings.OGIMG + filePathName[(filePathName.rindex('/')):-4] + '.jpg'
                # have this send the name of the file in augImg that matches the normal for now
                fs.delete(fileObj.name)

                context = {'message1': f'Confidence in Prediction: %.2f' % ((final_score1)),
                           'message': f'Predicted Label: %s' % (final_name1), 'ogImg': f'%s' % (ogImg),
                           'actMes': "View Activation maps", 'heatMes': "View Heat maps"}
                return render(request, 'testing.html', context)
        return render(request, 'testing.html')

    def plot_acc(request):
        covidCR_model_2 = pickle.load(open(settings.CHESTCR_MODEL))
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

    def heat_maps(request, file=None):
        DNN_layers = [layer.name for layer in ChestCR_model.layers]
        if request.method == 'POST':
            if os.path.exists(settings.HEAT_MAPS): shutil.rmtree(settings.HEAT_MAPS, ignore_errors=True)
            os.mkdir(settings.HEAT_MAPS)
            layer = request.POST.get('layers')
            fileObj = request.FILES['filePath']
            fs = FileSystemStorage()
            filePathName = fs.save("heatMaps/" + fileObj.name, fileObj)
            # filePathName = fs.save(fileObj.name, fileObj)
            filePathName = fs.url(filePathName)
            print(filePathName)
            test_image = '.' + filePathName
            print(test_image)
            input_test = Home.process_scan("", test_image)
            activations = get_activations(ChestCR_model, np.expand_dims(input_test, axis=0), layer)
            # input_test = Home.preprocess_image(img_path=test_image, model=covid_kaggle_model,
            # resize=(Image_Height, Image_Width))
            # activations = get_activations(covid_kaggle_model, input_test, layer)
            heatMapImgPath = display_heatmaps(activations, input_test, directory=r'./media/heatMaps', save=True)
            # print( heatMapImgPath)

            layer = "Viewing layer: " + layer
            key = "Key: Model valued areas that are closer to yellow more when determining its result"
            return render(request, 'heat_maps.html',
                          {"layer_name": layer, "DNN_layers": DNN_layers, 'filePathName': filePathName,
                           'HeatMapImgPath': heatMapImgPath, 'key': key})
        return render(request, 'heat_maps.html', {"DNN_layers": DNN_layers})

    def activation_maps(request, file=None):

        # just change path it is being displayed from
        DNN_layers = [layer.name for layer in ChestCR_model.layers]
        if request.method == 'POST':
            if os.path.exists(settings.ACTIVATION_MAPS): shutil.rmtree(settings.ACTIVATION_MAPS, ignore_errors=True)
            os.mkdir(settings.ACTIVATION_MAPS)
            layer = request.POST.get('layers')
            fileObj = request.FILES['filePath']
            fs = FileSystemStorage()

            filePathName = fs.save("actMaps/" + fileObj.name, fileObj)
            # filePathName = fs.save(fileObj.name, fileObj)
            # filePathName = "/" + filePathName

            filePathName = fs.url(filePathName)
            test_image = '.' + filePathName
            input_test = Home.process_scan("activation", test_image)
            activations = get_activations(ChestCR_model, np.expand_dims(input_test, axis=0), layer)
            # print(type(activations))
            # print(locals())
            # print(activations[layer])
            temp = activations[layer]
            # solo(temp[1][1])
            cur = temp[0][1]
            activationMapImgPath = display_activations(activations, directory=r'./media/actMaps', save=True)
            # possibly get activationmaps image and then divide it into pictures from there

            layer = "Viewing layer: " + layer
            key = "Key: Model valued areas that are closer to yellow more when determining its result"
            return render(request, 'activation_maps.html',
                          {"layer_name": layer, "DNN_layers": DNN_layers, 'filePathName': filePathName,
                           'ActivationMapImgPath': activationMapImgPath, 'key': key})
        return render(request, 'activation_maps.html', {"DNN_layers": DNN_layers})

    def shapley_values(request):
        # if request.method == 'POST':
        X_train = np.load(settings.DATA + 'xtrain.npy')
        X_test = np.load(settings.DATA + 'xtest.npy')
        background = X_train[np.random.choice(X_train.shape[0], 10, replace=False)]
        e = shap.DeepExplainer(ChestCR_model, np.expand_dims(background, axis=-1))
        X_test_expand = np.expand_dims(X_test, axis=-1)
        X_test_float = X_test_expand.astype(float)
        shap_values = e.shap_values(X_test_float[5:7])
        X_testplot_float = X_test.astype(float)
        # fig= shap.image_plot(shap_values, X_testplot_float[5:7])
        canvas = FigureCanvasTkAgg(shap.image_plot(shap_values, X_testplot_float[5:7], show='False'))
        response = HttpResponse(content_type='image/png')
        canvas.print_png(response)
        return response

    # return render(request, 'shapley_value.html')

    def display_opencv_image(request):
        image = np.zeros((500, 500, 3), dtype="uint8")
        cv2.line(image, (0, 0), (500, 500), (0, 0, 255), 5)

        cv2.imshow("Img", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return HttpResponse("Image Display complete")

    def shapely_value(request):
        # Load the image with Keras utilities
        img_path = 'data/sample.jpg'
        image = load_img(img_path)
        image_array = img_to_array(image)

        # Create a data generator for augmentation
        datagen = ImageDataGenerator(rotation_range=45)
        for batch in datagen.flow(np.expand_dims(image_array, axis=0), batch_size=1):
            rotated_image_array = batch[0]
            break
        rotated_image = array_to_img(rotated_image_array)
        rotated_path = 'data/rotated_sample.jpg'
        rotated_image.save(rotated_path)

        return render(request, 'monitor.html', {'image_path': '/data/rotated_sample.jpg'})

    def augmentations(request):
        files = os.listdir(settings.AUGIMG)
        if request.method == 'POST':
            image = request.POST.get('image')
            og = settings.OGIMG + settings.SEP + image
            aug = settings.AUGIMG + settings.SEP + image
            ogMes = "Original Image:"
            augMes = "Augmented Image:"
            return render(request, 'augmentations.html',
                          {"files": files, "og": og, "aug": aug, "ogMes": ogMes, "augMes": augMes})

        return render(request, 'augmentations.html', {"files": files})

    def test(request):
        return render(request, 'test.html')

    def jobs(request=None):

        # print(Home.track_jobs())
        data = Home.track_jobs()
        return render(request, 'jobs.html', {"data": data})