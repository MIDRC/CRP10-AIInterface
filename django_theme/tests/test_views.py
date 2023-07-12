from django.test import LiveServerTestCase
from selenium.webdriver.common.keys import Keys
import time
from selenium import webdriver
from selenium.webdriver.firefox.options import Options

class Hosttest(LiveServerTestCase):

    def testhomepage(self):
        driver =  webdriver.Chrome()
        driver.get('http://127.0.0.1:8000/')
        assert "MIDRC AI Interface for COVID" in driver.title

class LoginFormTest(LiveServerTestCase):

    def testform(self):
        driver = webdriver.Chrome()
        driver.get('http://127.0.0.1:8000/login/')
        time.sleep(5)
        user_name = driver.find_element('name', 'username')
        user_password = driver.find_element('name', 'password')
        time.sleep(5)
        submit = driver.find_element('id', 'submit')
        user_name.send_keys('ngorre')
        user_password.send_keys('Rajnikanthgrs@23')
        submit.send_keys(Keys.RETURN)
        time.sleep(5)
        message = "First value and second value are not equal !"
        self.assertEqual(driver.current_url,'http://127.0.0.1:8000/index', message)

