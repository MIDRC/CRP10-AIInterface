from django_theme.celery import app
from app.models import Tasks
from time import sleep
from app import views
from tensorflow import keras
import random
from django.conf import settings

normal_scan_path = settings.NORMAL_SCAN_PATH
abnormal_scan_path = settings.ABNORMAL_SCAN_PATH


@app.task(bind=True)
def process(self, job_name=None):
    b = Tasks(task_id=self.request.id, job_name=job_name)
    b.save()

    self.update_state(state='Dispatching', meta={'progress': '33'})
    sleep(random.randint(5, 10))

    self.update_state(state='Running', meta={'progress': '66'})
    sleep(random.randint(5, 10))
    self.update_state(state='Finishing', meta={'progress': '100'})
    sleep(random.randint(5, 10))

@app.task(bind=True)
def process_training(self, Epochs, LearningRate, Batchsize, Loss, Opt, job_name=None):
    b = Tasks(task_id=self.request.id, job_name=job_name)
    b.save()
    print(LearningRate)
    print(Loss)

    # sanitizing dropdowns
    if (Loss == 'BCE loss'):
        Loss = 'binary_crossentropy'
    elif (Loss == "CCE loss"):
        Loss = 'categorical_crossentropy'
    elif (Loss == "hinge loss"):
        Loss = 'hinge'
    elif (Loss == "MSLE loss"):
        Loss = 'msle'
    elif (Loss == "MAE loss"):
        Loss = 'mae'

    self.update_state(state='Pre-processing', meta={'progress': '33'})
    print("State updated to Pre-processing with progress 33")
    sleep(random.randint(5, 10))
    # update()

    #X_train, y_train, X_test, y_test, X_val, y_val = views.Home.pixelarray(Augment, normal_scan_path,abnormal_scan_path)
    # X_train = np.repeat(X_train[..., np.newaxis], 3, -1)
    # X_val = np.repeat(X_val[..., np.newaxis], 3, -1)

    # #simply for testing, do put your own xtrain either in media/data or change local_settings to reflect it
    # np.save(settings.DATA + 'xtrain.npy',X_train)
    # np.save(settings.DATA + 'xtest.npy',X_test)

    self.update_state(state='compiling', meta={'progress': '66'})
    sleep(random.randint(5, 10))
    print("State updated to compiling with progress 66")
    # update()

   # CRcl_model = views.Home.model2()

    # allows any optimizer on the dropdown to be used
    # if (Opt == "Adam"):
    #     CRcl_model.compile(loss=Loss, optimizer=keras.optimizers.Adam(learning_rate=LearningRate), metrics=["acc"], )
    # elif (Opt == "RMSProp"):
    #     CRcl_model.compile(loss=Loss, optimizer=keras.optimizers.RMSprop(learning_rate=LearningRate), metrics=["acc"], )
    # elif (Opt == "SGD"):
    #     CRcl_model.compile(loss=Loss, optimizer=keras.optimizers.SGD(learning_rate=LearningRate), metrics=["acc"], )
    # # MB-SGD  not in keras.optimizers or in any related packages
    # elif (Opt == "ADA grad"):
    #     CRcl_model.compile(loss=Loss, optimizer=keras.optimizers.Adagrad(learning_rate=LearningRate), metrics=["acc"], )
    # elif (Opt == "ADA delta"):
    #     CRcl_model.compile(loss=Loss, optimizer=keras.optimizers.Adadelta(learning_rate=LearningRate),
    #                        metrics=["acc"], )
    # NGA not in keras.optimizers, in evolutionary_keras.optimizers

    # sleep(random.randint(5, 10))

    self.update_state(state='fitting and training model', meta={'progress': '100'})
    sleep(random.randint(5, 100) + 10)
    # update()
    #CRcl_model.fit(X_train, y_train, epochs=Epochs, batch_size=Batchsize, validation_data=(X_val, y_val), )
    # update() #might not be needed here or needed somewhere else


def update():
    # j = json.dumps(views.Home.track_jobs(), indent=4)

    # with open("app/user_jobs.json", "w") as outfile:
    #     outfile.write(j)
    return (views.Home.jobs())
    # do i even need update or should I just call views.home.jobs?