from django import forms
#
# LOSS_CHOICES= [
#     ('binarycrossentropy', 'Binarycrossentropy'),
#     ('categoricalcrossentropy', 'Categoricalcrossentropy'),
#     ]
#
# OPTIMIZER_CHOICES =[('adam','ADAM'),
#                     ('rmsprop','RMSPROP'),
#                     ('sgd','SGD')]
#
# DNN_LAYER_CHOICES = [('conv2d','Conv2D'),('dropout','DROPOUT')]
#
# # creating a form
# class InputForm(forms.Form):
#     number_of_epochs = forms.IntegerField(max_value = 200)
#     batchsize = forms.IntegerField(max_value = 200)
#     learning_rate = forms.DecimalField()
#     loss_choices= forms.CharField(label='Choose the loss function:', widget=forms.Select(choices=LOSS_CHOICES))
#     optimizer_choices= forms.CharField(label='Choose the optimizer:', widget=forms.Select(choices=OPTIMIZER_CHOICES))
#
# class HeatmapForm(forms.Form):
#     dnn_layer_choices = forms.CharField(widget=forms.Select(choices=DNN_LAYER_CHOICES))


class JobForm(forms.Form):
    job_name = forms.CharField(label='Job name', required=True)
