#this is for DGX setup, which is on linux
make:
	# Following requires sudo/IT if erlang/rabbitmq are not installed
	apt-get install erlang-base -y #requires sudo/IT
	apt-get install rabbitmq-server -y --fix-missing
	
	# Following installs miniconda
	mkdir -p miniconda3
	wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda3/miniconda.sh
	bash miniconda3/miniconda.sh -b -u -p miniconda3
	rm -rf miniconda3/miniconda.sh
	# terminal must be restarted for this to take effect

rabbit:
	#Requires sudo/IT
	apt-get install erlang-base -y #requires sudo/IT
	apt-get install rabbitmq-server -y --fix-missing
	
conda:
	mkdir -p miniconda3
	wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda3/miniconda.sh
	bash miniconda3/miniconda.sh -b -u -p miniconda3
	rm -rf miniconda3/miniconda.sh
	# terminal must be restarted for this to take effect

env:
	conda create -n mai python=3.10.0
	# Now run 'conda activate mai'
	# followed by pip install requirements.txt

d:
	# must be in environment (run 'make mai' if one is not done)
	# this runs the django server, will ask you what GPU it should run on
	cd django_theme && python manage.py runserver

r:
	# must be in environment (run 'make mai' if one is not done)
	# this connects to the rabbitmq server, will ask you what GPU it should run on
	cd django_theme && celery -A django_theme worker -l info -P gevent