from django.shortcuts import render
from .models import News
import csv
from django.conf import settings
import statistics
import plotly.graph_objs as go
from plotly.offline import plot
from .forms import CSVUploadForm
from django.http import JsonResponse






def auto_download(request):
    if request.method == 'POST':
        company = request.POST.get('company')
        from selenium.webdriver import Chrome
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support import expected_conditions as EC
        from selenium.webdriver.support.ui import WebDriverWait
        import time
        from selenium.webdriver.common.keys import Keys
        from selenium.common.exceptions import NoSuchElementException, WebDriverException


        chrome_options = Options()
        chrome_options.add_experimental_option("prefs", {
            "download.prompt_for_download": True,
            "download.directory_upgrade": True,
            "safebrowsing.enabled": True
        })

        driver = Chrome(options=chrome_options)
        try:

            driver.get('https://nepsealpha.com/nepse-data')

            time.sleep(5)
            # Prompt the user to solve the CAPTCHA manually
            print("Please solve the CAPTCHA manually in the browser window...")
            input("Press Enter after you have solved the CAPTCHA and the page has loaded: ")
            try:
            # Locate the button element by its aria-label attribute
                button_element = driver.find_element(By.CSS_SELECTOR, 'button[aria-label="Dismiss sign-in info."]')

                # Click the button
                button_element.click()
            except NoSuchElementException:
                print("The button element was not found on the page.")
            except WebDriverException as e:
                print(f"An error occurred while trying to interact with the button: {e}")


            """ select_click = driver.find_element(By.CSS_SELECTOR, '#vue_app_content > div.page.page_margin_top > div > div > div > form > div > div > div:nth-child(4) > span > span.selection > span')
            select_click.click()

            select_input = driver.find_element(By.CSS_SELECTOR, 'body > span > span > span.select2-search.select2-search--dropdown > input')
            select_input.send_keys(company)
            select_input.send_keys(Keys.ENTER)

            start_date = driver.find_element(By.CSS_SELECTOR, '#vue_app_content > div.page.page_margin_top > div > div > div > form > div > div > div:nth-child(2) > input')
            #time.sleep(5)
            start_date.send_keys("10-10-2020")
            time.sleep(3) """
            
            """  #from here
            # Wait until the element is visible and interactable
            # Find the date input field
            date_input = driver.find_element(By.CSS_SELECTOR, 'input[type="date"]')
            time.sleep(5)
            # Set the date value in the input field
            date_input.send_keys('10-10-2020')
            #up to here
            time.sleep(20) """
            # Find the submit button by class name
            
            
            """ filter_button = driver.find_element(By.CSS_SELECTOR, '#vue_app_content > div.page.page_margin_top > div > div > div > form > div > div > div:nth-child(5) > button')
            filter_button.click()
            time.sleep(6)

            csv_button = driver.find_element(By.CSS_SELECTOR, '#result-table_wrapper > div.dt-buttons > button.dt-button.buttons-csv.buttons-html5.btn.btn-outline-secondary.btn-sm')
            csv_button.click()



            time.sleep(5)
            driver.quit()
            import os
            import subprocess

            # Get the user's download folder path
            download_folder = os.path.expanduser("~\\Downloads")

            # Open the download folder in Windows Explorer
            subprocess.Popen(f'explorer "{download_folder}"') """
            
        except:
             
            data ={
                'error': 1
            }
            
            return render(request,'data.html', data)
            

        return render(request,'data.html')

from django.http import JsonResponse, HttpResponseBadRequest
import pandas as pd

def predict(request):
    if request.method == 'POST' and 'csv_file' in request.FILES:
        model = request.POST.get('model')
        prediction_days = request.POST.get('numberInput')
        dropout_rate = request.POST.get('dropoutRate')

        try:
            prediction_days = int(prediction_days)
            dropout_rate = float(dropout_rate) if dropout_rate else 0.0
        except ValueError:
            return HttpResponseBadRequest("Invalid number of prediction days.")

        csv_file = request.FILES['csv_file']
        result_dict = {}

        lstm_result = gru_result = rf_result = None

        if model in ['LSTM', 'ALL']:
            from .lstm import lstm_model
            lstm_result = lstm_model(csv_file, prediction_days, dropout_rate)
            csv_file.seek(0)  # Reset the file pointer
            if lstm_result:
                result_dict['lstm_prediction'] = lstm_result[0].to_dict()

        if model in ['GRU', 'ALL']:
            from .gru import gru_model
            gru_result = gru_model(csv_file, prediction_days, dropout_rate)
            csv_file.seek(0)  # Reset the file pointer
            if gru_result:
                result_dict['gru_prediction'] = gru_result[0].to_dict()

        if model in ['RF', 'ALL']:
            from .rf import random_forest_model
            rf_result = random_forest_model(csv_file, prediction_days)
            if rf_result:
                result_dict['rf_prediction'] = rf_result[0].to_dict()

        if model == 'ALL':
            combined_result = {}
            if lstm_result:
                combined_result['date'] = lstm_result[0].index.tolist()
                combined_result['lstm_values'] = lstm_result[0]['close_price'].tolist()
            if gru_result:
                combined_result['gru_values'] = gru_result[0]['close_price'].tolist()
            if rf_result:
                combined_result['rf_values'] = rf_result[0]['close_price'].tolist()

            result_dict['combined'] = combined_result

        return JsonResponse({'data': result_dict})

    return render(request, 'predict.html')

def data_download(request):
    return render(request, 'data.html')








def visualize_csv_form(request):
    if request.method == 'POST':
        form = CSVUploadForm(request.POST, request.FILES)
        if form.is_valid():
            csv_file = request.FILES['csv_file']
            reader = csv.reader(csv_file.read().decode('utf-8').splitlines())
            header = next(reader)  # Skip the header row
            data = list(reader)

            # Extract column data
            dates = [row[1] for row in data]  # Assuming the date column is at index 1
            close_prices = [float(row[5]) for row in data]  # Assuming the close price column is at index 5

            # Calculate statistical data
            minimum = min(close_prices)
            maximum = max(close_prices)
            average = statistics.mean(close_prices)
            variance = statistics.variance(close_prices)
            median = statistics.median(close_prices)

            chart_data = go.Scatter(x=dates, y=close_prices, mode='lines', name='Close Prices')
            layout = go.Layout(title='Close Prices Over Time', xaxis=dict(title='Date'), yaxis=dict(title='Close Price'))
            fig = go.Figure(data=[chart_data], layout=layout)
            plot_div = plot(fig, output_type='div')

            return render(request, 'visualization.html', {'form': form, 'plot_div': plot_div, 'minimum': minimum, 'maximum': maximum, 'average': average, 'variance': variance, 'median': median})
    else:
        form = CSVUploadForm()

    return render(request, 'visualization.html', {'form': form})













def get_driver():
    from selenium.webdriver import Chrome
    from selenium.webdriver.chrome.options import Options

    chrome_options = Options()
    chrome_options.add_argument("--headless")

    driver = Chrome(options=chrome_options)
    return driver

# Create your views here.

def index(request):
    return render(request,'index.html')


def finding(request):
    return render(request,'finding.html')

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json

@csrf_exempt
def chatbot_question(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            question = data.get('question', '').strip()  # Trim any extra spaces

            print(f"Received question: {question}")  # Log question for debugging

            # Define answers for predefined questions
            answers = {
                "Q. What is stock?": "A stock is a type of security that signifies proportionate ownership in the issuing corporation.",
                "Q. What is R2 score and RMSE score?": "R2 score is a statistical measure that represents the proportion of variance for a dependent variable that's explained by an independent variable. RMSE is the square root of the average of squared differences between prediction and actual observation.",
                "Q. Which models are used here?": "The models used here are LSTM, GRU, and Random Forest.",
                "Q. Which model has shown best result after comparison?": "After comparison, the GRU model has shown the best result.",
                "Q. Team members": "Our team members are Anurag Paudel, Avishek Hada, Habi Pyatha, Sujan Dhoj Karki",
                "Q. Where can I get datas from?": 'You can obtain data for 5 years from <a href="https://nepsealpha.com/nepse-data" target="_blank">Nepse Alpha</a> or go to the visualization page and click data.',
            }

            # Get the appropriate answer or a default response
            answer = answers.get(question, "Sorry, I don't understand the question.")
            
            print(f"Answer: {answer}")  # Log answer for debugging

            return JsonResponse({'answer': answer})

        except json.JSONDecodeError:
            return JsonResponse({'answer': 'Invalid JSON format.'}, status=400)
    
    return JsonResponse({'answer': 'Invalid request method.'}, status=400)
