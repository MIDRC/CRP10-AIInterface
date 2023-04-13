# tasks.py would contain the model training tasks which are used to track, monitor while performing training and
# the resultant is utilized to display on the progress bars

from django_theme.celery import app
from app.models import Tasks
from time import sleep
import random

normal_scan_path = r"M:\dept\Dept_MachineLearning\Staff\ML Engineer\Naveena Gorre\Datasets\Covid_MIDRC\Covid_Classification\Covid_negative"
abnormal_scan_path = r"M:\dept\Dept_MachineLearning\Staff\ML Engineer\Naveena Gorre\Datasets\Covid_MIDRC\Covid_Classification\Covid_positive"

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

def process_training(self,Epochs,LearningRate, Batchsize,job_name =None):

    b = Tasks(task_id=self.request.id,job_name=job_name)
    b.save()
    print(LearningRate)
    self.update_state(state='Pre-processing', meta={'progress': '33'})
    #X_train, y_train, X_test, y_test, X_val, y_val = views.Home.pixelarray(normal_scan_path, abnormal_scan_path)
    sleep(random.randint(5, 10))
    self.update_state(state='compiling', meta={'progress': '66'})
    #CRcl_model = views.Home.model2()
    #CRcl_model.compile(loss="binary_crossentropy",optimizer=keras.optimizers.Adam(learning_rate=LearningRate),metrics=["acc"],)
    sleep(random.randint(5, 10))
    self.update_state(state='fitting and training model', meta={'progress': '100'})
    #CRcl_model.fit(X_train, y_train, epochs=Epochs, batch_size=Batchsize, validation_data=(X_val, y_val),)
    sleep(random.randint(5, 10))


