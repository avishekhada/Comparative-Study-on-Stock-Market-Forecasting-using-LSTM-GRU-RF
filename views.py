from django.shortcuts import render
from .models import News
import csv
from django.conf import settings
import statistics
import plotly.graph_objs as go
from plotly.offline import plot
from .forms import CSVUploadForm
from django.http import JsonResponse


def predict(request):
    from django.http import HttpResponseBadRequest
    if request.method == 'POST' and request.FILES['csv_file']:
        
        model = request.POST.get('model')
        csv_file = request.FILES['csv_file']

        prediction_days=request.POST.get('numberInput')
        dropout_rate=request.POST.get('dropoutRate')
        #print("value is passed check",prediction_days)
        #print("POST data:", request.POST)
        #prediction_day=int(prediction_days)
        try:
            prediction_days = int(prediction_days)
            #dropout_rate=float(dropout_rate)
            if dropout_rate is None:
                dropout_rate = 0.0
            else:
                dropout_rate = float(dropout_rate)
            print("value is passed",prediction_days)
            print("dropout rate:",dropout_rate)
        except ValueError:
            return HttpResponseBadRequest("Invalid number of prediction days.")
        print(model)
        if(model == 'LSTM'):
            from .lstm import lstm_model
            csv_file = request.FILES['csv_file']
            result = lstm_model(csv_file,prediction_days,dropout_rate)
            
            
            if result is not None and len(result) >= 10:
                result_dict = {
                    'prediction': result[0].to_dict() if result[0] is not None else {},
                    'train_rmse': result[1],
                    'test_rmse': result[2],
                    'train_r2': result[3],
                    'test_r2': result[4],
                    'train_mae':result[5],
                    'test_mae':result[6],
                    'plot_div': result[7],
                    'plot_div_1': result[8],
                    'plot_div_2': result[9],
                    'plot_div_3': result[10],
                }
                
            else:
                # Handle the error case
                result_dict = {
                    'prediction': {},
                    'train_rmse': None,
                    'test_rmse': None,
                    'train_r2': None,
                    'test_r2': None,
                    'train_mae':None,
                    'test_mae':None,
                    'plot_div': '',
                    'plot_div_1': '',
                    'plot_div_2': '',
                    'plot_div_3': ''

                }
        
        elif(model == 'GRU'):
            from .gru import gru_model
            csv_file = request.FILES['csv_file']
            result = gru_model(csv_file,prediction_days,dropout_rate)
           
            if result is not None and len(result) >= 10:
                result_dict = {
                    'prediction': result[0].to_dict() if result[0] is not None else {},
                    'train_rmse': result[1],
                    'test_rmse': result[2],
                    'train_r2': result[3],
                    'test_r2': result[4],
                    'train_mae':result[5],
                    'test_mae':result[6],
                    'plot_div': result[7],
                    'plot_div_1': result[8],
                    'plot_div_2': result[9],
                    'plot_div_3': result[10],
                }
                
            else:
                # Handle the error case
                result_dict = {
                    'prediction': {},
                    'train_rmse': None,
                    'test_rmse': None,
                    'train_r2': None,
                    'test_r2': None,
                    'train_mae':None,
                    'test_mae':None,
                    'plot_div': '',
                    'plot_div_1': '',
                    'plot_div_2': '',
                    'plot_div_3': ''
                }
        
        elif(model == 'RF'):
            from .rf import random_forest_model
            csv_file = request.FILES['csv_file']
            result = random_forest_model(csv_file,prediction_days)
            
            if result is not None and len(result) >= 10:
                result_dict = {
                    'prediction': result[0].to_dict() if result[0] is not None else {},
                    'train_rmse': result[1],
                    'test_rmse': result[2],
                    'train_r2': result[3],
                    'test_r2': result[4],
                    'train_mae':result[5],
                    'test_mae':result[6],
                    'plot_div': result[7],
                    'plot_div_1': result[8],
                    'plot_div_2': result[9],
                    'plot_div_3': result[10],
                    
                }
                
            else:
                # Handle the error case
                result_dict = {
                    'prediction': {},
                    'train_rmse': None,
                    'test_rmse': None,
                    'train_r2': None,
                    'test_r2': None,
                    'train_mae':None,
                    'test_mae':None,
                    'plot_div': '',
                    'plot_div_1': '',
                    'plot_div_2': '',
                    'plot_div_3': ''
                }
        elif(model=='ALL'):
            from .lstm import lstm_model
            from .gru import gru_model
            from .rf import random_forest_model
            csv_file.seek(0)
            lstm_result = lstm_model(csv_file, prediction_days, dropout_rate)
            if lstm_result is not None and len(lstm_result) >= 10:
                result_dict1 = {
                    'prediction': lstm_result[0].to_dict() if lstm_result[0] is not None else {},
                    'train_rmse': lstm_result[1],
                    'test_rmse': lstm_result[2],
                    'train_r2': lstm_result[3],
                    'test_r2': lstm_result[4],
                    'train_mae':lstm_result[5],
                    'test_mae':lstm_result[6],
                    'plot_div': lstm_result[7],
                    'plot_div_1': lstm_result[8],
                    'plot_div_2': lstm_result[9],
                    'plot_div_3': lstm_result[10],
                }
                
            else:
                # Handle the error case
                result_dict1 = {
                    'prediction': {},
                    'train_rmse': None,
                    'test_rmse': None,
                    'train_r2': None,
                    'test_r2': None,
                    'train_mae':None,
                    'test_mae':None,
                    'plot_div': '',
                    'plot_div_1': '',
                    'plot_div_2': '',
                    'plot_div_3': ''
                }
            csv_file.seek(0)
            gru_result = gru_model(csv_file, prediction_days, dropout_rate)

            if gru_result is not None and len(gru_result) >= 10:
                result_dict2 = {
                    'prediction': gru_result[0].to_dict() if gru_result[0] is not None else {},
                    'train_rmse': gru_result[1],
                    'test_rmse': gru_result[2],
                    'train_r2': gru_result[3],
                    'test_r2': gru_result[4],
                    'train_mae':gru_result[5],
                    'test_mae':gru_result[6],
                    'plot_div': gru_result[7],
                    'plot_div_1': gru_result[8],
                    'plot_div_2': gru_result[9],
                    'plot_div_3': gru_result[10],
                }
                
            else:
                # Handle the error case
                result_dict2 = {
                    'prediction': {},
                    'train_rmse': None,
                    'test_rmse': None,
                    'train_r2': None,
                    'test_r2': None,
                    'train_mae':None,
                    'test_mae':None,
                    'plot_div': '',
                    'plot_div_1': '',
                    'plot_div_2': '',
                    'plot_div_3': ''
                }
            csv_file.seek(0)
            rf_result = random_forest_model(csv_file, prediction_days)

            if rf_result is not None and len(rf_result) >= 10:
                result_dict3 = {
                    'prediction': rf_result[0].to_dict() if rf_result[0] is not None else {},
                    'train_rmse': rf_result[1],
                    'test_rmse': rf_result[2],
                    'train_r2': rf_result[3],
                    'test_r2': rf_result[4],
                    'train_mae':rf_result[5],
                    'test_mae':rf_result[6],
                    'plot_div': rf_result[7],
                    'plot_div_1': rf_result[8],
                    'plot_div_2': rf_result[9],
                    'plot_div_3': rf_result[10],
                    
                }
                
            else:
                # Handle the error case
                result_dict3= {
                    'prediction': {},
                    'train_rmse': None,
                    'test_rmse': None,
                    'train_r2': None,
                    'test_r2': None,
                    'train_mae':None,
                    'test_mae':None,
                    'plot_div': '',
                    'plot_div_1': '',
                    'plot_div_2': '',
                    'plot_div_3': ''
                }
            # print(result_dict1)
            # print(result_dict2)
            # print(result_dict3)
            result_dict = {
                'LSTM': result_dict1,
                'GRU': result_dict2,
                'RF': result_dict3
            }
        return JsonResponse({'data': result_dict})
    
    return render(request, 'predict.html')
    
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
            minimum = round(min(close_prices), 2)
            maximum = round(max(close_prices), 2)
            average = round(statistics.mean(close_prices), 2)
            variance = round(statistics.variance(close_prices), 2)
            median = round(statistics.median(close_prices), 2)

            chart_data = go.Scatter(x=dates, y=close_prices, mode='lines', name='Close Prices')
            layout = go.Layout(title='Close Prices Over Time', xaxis=dict(title='Date'), yaxis=dict(title='Close Price'))
            fig = go.Figure(data=[chart_data], layout=layout)
            plot_div = plot(fig, output_type='div')

            return render(request, 'visualization.html', {'form': form, 'plot_div': plot_div, 'minimum': minimum, 'maximum': maximum, 'average': average, 'variance': variance, 'median': median})
    else:
        form = CSVUploadForm()

    return render(request, 'visualization.html', {'form': form})


# Create your views here.

def index(request):
    return render(request,'index.html')


def finding(request):
    return render(request,'finding.html')