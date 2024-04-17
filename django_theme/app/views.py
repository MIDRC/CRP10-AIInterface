# Original Author: Naveena Gorre
# Email address: naveena.gorre@moffitt.org
# views.py is the main backend code populated with all the python functions needed to run the frontend templates.

# All the necessary imports/python packages needed to run the DL models, pre-processing steps as well as visulization
# and interpretability

import PIL,shap, cv2, pydicom, pickle,re, io,os
from django.shortcuts import render,redirect,HttpResponse
import subprocess
#import slideio
from json import dumps
from django.core.files.storage import FileSystemStorage
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img, img_to_array
from django.views.generic import TemplateView
from django.http import HttpResponse
from django.contrib.auth.models import User
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.decorators import login_required
from keras.models import load_model
from keract import get_activations
from keract import display_activations,display_heatmaps
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.layers import Input, Dense,Conv2D, BatchNormalization, Activation, Flatten
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

# Selenium setup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

# Create your views here.

# Here all the models trained as well as untrained are loaded onto the environment

MODEL = load_model('.\\models\\covidnet.hdf5')
ChestCR_model = load_model(r'C:\Users\4472829\PycharmProjects\Jupyter_notebook\saved_models\covidCRnet.hdf5')
covidCR_model_2 = pickle.load(open('.\\models\\finetuning_imagenet_hpt', "rb"))
covid_kaggle_model = load_model('.\\models\\\covid_kagglenet.hdf5')

Image_Height, Image_Width = 150,150 
BATCH_SIZE, EPOCHS = 5,5

# Defining class based views here: First class is to define the functions for login and signup of the API

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


# Home is the main class based view which would have all the major functions of backend useful
# to run the frontend templates

class Home(TemplateView):

    def login_base(request):
        return render(request, 'base_auth.html')

    @login_required
    def index(request):
         context = {'a': 1}
         return render(request, 'index.html')

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

# This function performs model training as well as displays the progress bars of various steps involved in training
    def run_training(request):
       if request.method == 'POST':
            Epochs = int(request.POST.get('epochVal'))
            Batchsize = int(request.POST.get('batchsizeVal'))
            LearningRate = float(request.POST.get('learnrateVal'))
            loss = request.POST.get('lossVal')
            optimizer = request.POST.get('optimizerVal')
            model_input = request.POST.get('vggVal')
            aug_value = request.POST.get('augVal')
            if model_input == "Fine tuning":
                process_training.delay(Epochs,LearningRate,Batchsize,job_name = model_input)
                return render(request, 'training.html', context={'message': f'You have chosen {model_input}'})

            elif model_input == "Training from scratch":
                process_training.delay(job_name=model_input)
                return render(request, 'training.html', context={'message': f'You have chosen {model_input}'})
       return render(request, "training.html")


# Function which would track the jobs from the resultant obtained via return of track_jobs function
# once the train option is submitted on the training UI
    @require_GET
    def monitor_training(request):
        info = Home.track_jobs()
        return render(request, 'monitor.html', context={'info': info})

    @staticmethod
    def track_jobs():
        entries = Tasks.objects.all()
        information = []
        for item in entries:
            progress = 100
            result = AsyncResult(item.task_id)
            if isinstance(result.info, dict):
                progress = result.info['progress']
            information.append([item.job_name, result.state,
                                    progress, item.task_id])
        return information

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

# Idea behind this function is one chosen it would re-direct to Jupyter notebooks and integrate it with the
# API environment so that it would be helpful for advanced users to not only use the available options/models
# but also write/customize from the code
    # To-do
    def jupyter_notebook(request):
       if request.method == 'POST':
            b = subprocess.check_output("jupyter-lab list".split()).decode('utf-8')
            if "9999" not in b:
                a = subprocess.Popen("jupyter-lab  --no-browser --port 9999".split())
            start_time = time.time()
            unreachable_time = 10
            while "9999" not in b:
                timer = time.time()
                elapsed_time = timer - start_time
                b = subprocess.check_output("jupyter-lab list".split()).decode('utf-8')
                if "9999" in b:
                    break
                if elapsed_time > unreachable_time:
                    return HttpResponse("Unreachable")
            path = b.split('\n')[1].split('::', 1)[0]
            print(path)
            return render(request, path)
        # You can here add data to your path if you want to open file or anything
       return render(request, "jupyter_notebook.html")

# This function would pre-process and obtain the pixel array from the dicom image, utilized to train the
    #Chest X-ray classification model (VGG19)
    def process_scan(filepath):
        scan = pydicom.read_file(str(filepath))
        scan = scan.pixel_array
        scan = cv2.resize(scan, (224, 224))
        return scan

    # This function would pre-process and obtain the image array from the pathology slide, utilized to train the
    # pathology MDS classification model
    def process_slide(filepath):
        slide = slideio.open_slide(filepath,'SVS')
        scene = slide.get_scene(0)
        image = scene.read_block(size=(150, 150))
        input_test = np.array(image)
        return input_test

    # scanpath function would return all the filenames with the full path/location of the files located
    # in a specific folder
    def scanpath(Base_path):
       scans = []
       for root, dirs, files in os.walk(Base_path):
           for fname in files:
               scans.append(os.path.join(root, fname))
       return scans

    # pixel array function would return the train, test, validation split of the data by taking the input paths of
    # data and then using the functions - scanpath, process_scan pre-process and obtain the input image arrays needed
    # to train the VGG19 model
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

    # model2 would return a pre-trained VGG19 model with additional classification head,
    # this model is utilized for fine-tuning purposes
    def model2(n_classes=2, input_shape=(224, 224, 1)):
       vgg_model = VGG19(include_top=False, weights=None, input_shape=input_shape)
       flat = Flatten()(vgg_model.layers[-1].output)
       proj = Dense(1024, activation='relu')(flat)
       soft = Dense(1, activation='sigmoid')(proj)
       model = Model(inputs=vgg_model.inputs, outputs=soft)
       return model

    #training_model function would perform the model training utilizing the parameters chosen from frontend (training.html)
    def training_model(request):
        if request.method == 'POST':
            Epochs = int(request.POST.get('epochVal'))
            batchsize = int(request.POST.get('batchsizeVal'))
            learningrate = float(request.POST.get('learnrateVal'))
            loss = request.POST.get('lossVal')
            optimizer = request.POST.get('optimizerVal')
            model_input = request.POST.get('vggVal')
            if model_input == "Fine tuning":
                # load data paths, process scans to obtain pixel array and generate labels
                print('You have chosen:',model_input,",extracting the pixel array of dicom images to train the model...")
                X_train, y_train, X_test, y_test, X_val, y_val = Home.pixelarray(normal_scan_path, abnormal_scan_path)
                CRcl_model = Home.model2()
                CRcl_model.compile(loss="binary_crossentropy",optimizer=keras.optimizers.Adam(learning_rate=0.001),metrics=["acc"],)
                CRcl_model.fit(X_train,y_train,epochs=5,batch_size=10,validation_data=(X_val, y_val),)

            context = ChestCR_model.history['accuracy']
            return render(request,'training.html',{"context": context,'epochs':Epochs})
        return render(request,"training.html")
    
    def removespaces(file_with_space):
        return os.rename(file_with_space,file_with_space.replace(' ', '_'))

    # Testing function to predict the image labels after the model is trained.
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

    # plot_acc function returns the acc, loss plots of the trained model
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

    # preprocess_image would perform pre-processing for the covid kaggle model and re-size
    # it to perform testing, visualization
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

    # heat_maps and activation_maps are utilized to perform interpretability/visualization of the black box AI algorithms.
    # Here we utilize a framework called keract and the user would have an option to choose an input image as well as a DNN layer
    # which would return a heatmap, activation map
    def heat_maps(request):
        DNN_layers = [layer.name for layer in ChestCR_model.layers]
        if request.method == 'POST':
            layer = request.POST.get('layers')
            fileObj = request.FILES['filePath']
            fs = FileSystemStorage()
            filePathName = fs.save(fileObj.name, fileObj)
            filePathName = fs.url(filePathName)
            test_image = '.' + filePathName
            #input_test_array = Home.process_slide(test_image)
            input_test = Home.process_scan(test_image)
            activations = get_activations(ChestCR_model, np.expand_dims(input_test, axis=0), layer)
            heatMapImgPath = display_heatmaps(activations, input_test, directory=r'./media/', save=True)
            return render(request, 'heat_maps.html',
                      {"layer_name": layer, "DNN_layers": DNN_layers, 'filePathName': filePathName,
                       'HeatMapImgPath':  heatMapImgPath})
        return render(request, 'heat_maps.html', {"DNN_layers": DNN_layers})
    
    def activation_maps(request):
        ChestCR_model_DNN_layers = [layer.name for layer in ChestCR_model.layers]
        covid_kaggle_model_DNN_layers = [layer.name for layer in covid_kaggle_model.layers]
        DNN_layers = ChestCR_model_DNN_layers + covid_kaggle_model_DNN_layers
        dnnLayers = {
            'ChestCR_model': ChestCR_model_DNN_layers,
            'covid_kaggle_model': covid_kaggle_model_DNN_layers,
            'DNN_layers': DNN_layers
        }
        ChestCR_model_Json = {
            'ChestCR_model': ChestCR_model_DNN_layers
        }
        dataJSON = dumps(dnnLayers)
        ChestCR_dataJSON = dumps(dnnLayers)
        if request.method == 'POST':
            layer = request.POST.get('layers')
            fileObj=request.FILES['filePath']
            fs = FileSystemStorage()
            filePathName = fs.save(fileObj.name, fileObj)
            filePathName = fs.url(filePathName)
            test_image = '.' + filePathName
            input_test = Home.process_scan(test_image)

            # If Model1 == covid_kaggle_model then run and generate the activation map
            if request.POST.get('models') == "ChestCR_model":
                activations = get_activations(ChestCR_model, np.expand_dims(input_test, axis=0), layer)
                activationMapImgPath = display_activations(activations, directory=r'./media/', save=True)
                print(activationMapImgPath)
                return render(request,'activation_maps.html',
                              {"layer_name": layer, "ChestCR_model_DNN_layers": ChestCR_model_DNN_layers, "DNN_layers": ChestCR_dataJSON, 'filePathName':filePathName,
                                                              'ActivationMapImgPath': activationMapImgPath})
            else:
                activations = get_activations(covid_kaggle_model, np.expand_dims(input_test, axis=0), layer)
                activationMapImgPath = display_activations(activations, directory=r'./media/', save=True)
                print(activationMapImgPath)
                return render(request, 'activation_maps.html',
                              {"layer_name": layer, "DNN_layers": covid_kaggle_model_DNN_layers,
                               'filePathName': filePathName,
                               'ActivationMapImgPath': activationMapImgPath})


        return render(request, 'activation_maps.html', {"DNN_layers": dataJSON})

    # shapley_values function is utilized to obtain shap plot for the images that user wanted to interpret
    def shapley_values(request):
        if request.method == 'POST':
            X_train = np.load(r'C:\Users\4472829\PycharmProjects\Pre_run_numpyarrays\xtrain.npy')
            X_test = np.load(r'C:\Users\4472829\PycharmProjects\Pre_run_numpyarrays\xtest.npy')
            background_samples = int(request.POST.get('trainsamplesVal'))
            background = X_train[np.random.choice(X_train.shape[0], background_samples, replace=False)]
            e = shap.DeepExplainer(ChestCR_model, np.expand_dims(background, axis=-1))
            X_test_expand = np.expand_dims(X_test, axis=-1)
            X_test_float = X_test_expand.astype(float)

            # Write an if condition here if the chosen values are in the range then plot the shap plot if not throw
            # a warning stating as a pop-up you have to chose between (0-10 etc)
            model_preds_initial = int(request.POST.get('modelVal1'))
            model_preds_final = int(request.POST.get('modelVal2'))
            shap_values = e.shap_values(X_test_float[model_preds_initial:model_preds_final])
            X_testplot_float = X_test.astype(float)
            shap.image_plot(shap_values, X_testplot_float[model_preds_initial:model_preds_final], show=False)
            plt.savefig('./media/shap.png')
            plt.close()
            image_path = r'./media/shap.png'
            return render (request, 'shapley_value.html', {'shapPath':image_path})
        return render(request, 'shapley_value.html')

    def interactive_dropdowns(request):
        if request.method=='POST':
            print(render,'interactive_dropdowns.html')
        return render(request, 'interactive_dropdowns.html')