from django.test import LiveServerTestCase
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from pathlib import Path 
import time, os
from django.conf import settings
#take in command line arg and have that direct to a test
path= str(Path.cwd())
# path =path.replace('django_theme', '')+ 'testing\\'
# UNZIP TESTING.ZIP IN THE MAI_CARE FOLDER, ONE LEVEL UP FROM TESTS.PY


options = webdriver.ChromeOptions()
options.binary_location= settings.CHROME
options.add_experimental_option('useAutomationExtension', False)

'''
class base(LiveServerTestCase):
  	
	def testform(self):
		options = webdriver.ChromeOptions()
		options.binary_location= chromeBin
		options.add_experimental_option('useAutomationExtension', False)
		#driver = webdriver.Chrome(options=options, executable_path=path+ str({path}))
		driver = webdriver.Chrome(options=options, executable_path=settings.CDRIVER)
		driver.get('http://127.0.0.1:8000/')

		
		assert "Moffitt" in driver.title

class login(LiveServerTestCase):

	def testform(self):
		options = webdriver.ChromeOptions()
		options.binary_location= chromeBin
		options.add_experimental_option('useAutomationExtension', False)
		driver = webdriver.Chrome(options=options, executable_path=settings.CDRIVER)
		driver.get('http://127.0.0.1:8000/login/')

		assert "Login" in driver.title
'''

class setup(LiveServerTestCase):
	
	def testform(self):
		driver = webdriver.Chrome(options=options, executable_path=settings.CDRIVER)
		driver.get('http://127.0.0.1:8000/')
		time.sleep(1) #allows everything to load
		#btn = driver.find_element_by_xpath('//*[@id="features"]/div/div[1]/div[3]/div[1]/a')
		driver.find_element_by_xpath('//*[@id="features"]/div/div[1]/div[3]/div[1]/a').click()
		assert 'http://127.0.0.1:8000/login_base' in driver.current_url 
		
		#divide each assert into differnt def

		driver.find_element_by_xpath('/html/body/nav/div/form/a[1]').click()
		assert 'http://127.0.0.1:8000/login' in driver.current_url
		
		username = driver.find_element_by_xpath('//*[@id="id_username"]')
		password =  driver.find_element_by_xpath('//*[@id="id_password"]')
		login = driver.find_element_by_xpath('/html/body/div/div/div/div/form/button')
		
		username.send_keys('admin') #TEMP admin, admin is local as of now so supply your own info
		password.send_keys('dogberto') #read above
		
		login.send_keys(Keys.RETURN)
		assert 'http://127.0.0.1:8000/index' in driver.current_url
		print("logging into mai-care is functional!")