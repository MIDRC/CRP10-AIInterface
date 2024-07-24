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
    #print(f"Augment in tasks: {augment}")

    self.update_state(state='Pre-processing', meta={'progress': '33'})
    print("State updated to Pre-processing with progress 33")
    sleep(random.randint(5, 10))

    self.update_state(state='compiling', meta={'progress': '66'})
    sleep(random.randint(5, 10))
    print("State updated to compiling with progress 66")

    # Assume CRcl_model is defined elsewhere and use it here
    self.update_state(state='fitting and training model', meta={'progress': '100'})
    sleep(random.randint(5, 100) + 10)
    #sleep(random.randint(5, 10))
    print("State updated to fitting and training model with progress 100")

def update():
    # j = json.dumps(views.Home.track_jobs(), indent=4)

    # with open("app/user_jobs.json", "w") as outfile:
    #     outfile.write(j)
    return (views.Home.jobs())
    # do i even need update or should I just call views.home.jobs?