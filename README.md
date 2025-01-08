**MIDRC CRP 10 - Visualization and Explainability of machine intelligence for prognosis and monitoring therapy** (https://www.midrc.org/midrc-collaborating-research-projects/project-one-crp10)

**Development Team**: Naveena Gorre (naveena.gorre@moffitt.org) 

**Problem Definition**: The main idea of CRP10AI interface is to use it as an interface for easy prototyping and testing of AI algorithms.

**Modality**: CT, X-ray, Clinical Data

**Requirements**: (Python, Tensorflow Keras API, OpenCV, Numpy, scikitlearn, pytorch)

**Repo Content Description**: The repo contains python/.py files, css, html and javascript. Currently the use case implemented here has covd image classification. HTML files are listed under templates and it's corresponsng css and Javascript are in css and js folder respectively. Views.py is the main python file which has all the backend class based views/functions to perform the operations. To run the repo clone the repo using git clone or download the whole folder and make the folder containing manage.py as the current working directory and run python manage.py runserver which launches the API as your localhost, copy and paste this url onto your webbrowser and the API is launched.

**Example Commands**: python manage.py runserver

**References**:

1)For information on MIDRC GitHub documentation and best practices, please see https://midrc.atlassian.net/wiki/spaces/COMMITTEES/pages/672497665/MIDRC+GitHub+Best+Practices

2)Moffitt Cancer Center and University of Chicago.

3)Lessons learned paper - https://doi.org/10.1117/1.JMI.8.S1.010902 
