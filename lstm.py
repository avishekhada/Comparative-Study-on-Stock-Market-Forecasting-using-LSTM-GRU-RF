

#from here
# -*- coding: utf-8 -*-
"""
Created on Wed May 22 17:44:52 2024

@author: Habi
"""



def lstm_model(company,prediction_days,dropout_rate):
    import pandas as pd
    from tensorflow.keras.callbacks import EarlyStopping
    import numpy as np
    from sklearn.preprocessing import MinMaxScaler
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    import matplotlib.pyplot as plt
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    import plotly.graph_objs as go
    from plotly.offline import plot
    from tensorflow.keras.optimizers import AdamW
    from tensorflow.keras.losses import Huber
    from tensorflow.keras.optimizers import SGD
    from tensorflow.keras.losses import LogCosh
    from tensorflow.keras.optimizers import Nadam

    prediction_days=prediction_days
    time_steps = 7
    def prepare_data(data, time_steps):
        X, y = [], []
        for i in range(len(data) - time_steps - prediction_days):
            X.append(data[i:i + time_steps, 0])
            y.append(data[i + time_steps:i + time_steps + prediction_days, 0])
        return np.array(X), np.array(y)

    try:
        # Read the CSV file
        df = pd.read_csv(company)

        # Convert the Date column to a datetime object
        df['Date'] = pd.to_datetime(df['Date'])

        # Sort the dataframe by date
        df = df.sort_values('Date')

        # Convert '--' to 0 in the 'Percent Change' column and handle non-numeric values
        df['Percent Change'] = pd.to_numeric(df['Percent Change'].replace('--', 0), errors='coerce').fillna(0)

        # Convert columns to float
        df['Open'] = df['Open'].astype(float)
        df['High'] = df['High'].astype(float)
        df['Low'] = df['Low'].astype(float)
        df['Close'] = df['Close'].astype(float)

        # Extract the 'Close' column for prediction
        data = df['Close'].values.reshape(-1, 1)

        # Scale the data using Min-Max Scaler
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)

        # Define the training and testing data sizes
        train_size = int(len(scaled_data) * 0.7)
        test_size = len(scaled_data) - train_size

        # Split the data into training and testing sets
        train_data = scaled_data[:train_size, :]
        test_data = scaled_data[train_size:, :]

        # Define the number of time steps
        #time_steps = 7

        # Prepare the training data
        X_train, y_train = prepare_data(train_data, time_steps)

        # Prepare the testing data
        X_test, y_test = prepare_data(test_data, time_steps)

        # Reshape the data for LSTM (samples, time_steps, features)
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

        # Define and compile the LSTM model
        model = Sequential()
        model.add(LSTM(units=100, return_sequences=True, input_shape=(time_steps, 1
        )))
        model.add(Dropout(dropout_rate))
        model.add(LSTM(units=100, return_sequences=True))
        model.add(Dropout(dropout_rate))
        model.add(LSTM(units=100))
        model.add(Dense(units=prediction_days))  # Output layer with 7 units for predicting 7 days ahead
        model.summary()

        model.compile(optimizer='adam', loss='mean_squared_error') #0.89
        # model.compile(optimizer='rmsprop', loss='mean_absolute_error')
        # model.compile(optimizer='rmsprop', loss='mean_absolute_error')#this is not good

        # model.compile(optimizer=SGD(momentum=0.9), loss='mean_absolute_percentage_error')
        # model.compile(optimizer='adam', loss=LogCosh()) # 0.9
        # model.compile(optimizer=Nadam(), loss='mean_squared_logarithmic_error')
        # Fit the model
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=32, callbacks=[early_stopping])
        # history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=32)

        # Plot the training and validation loss
        """ plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training and Validation Loss')
        plt.show() """
        

        # Make predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Inverse transform the predictions
        train_predict = scaler.inverse_transform(train_predict)
        test_predict = scaler.inverse_transform(test_predict)
        y_train = scaler.inverse_transform(y_train)
        y_test = scaler.inverse_transform(y_test)

        # Plot training predictions
        y_train_one_day = y_train[:, 0]
        train_predict_one_day = train_predict[:, 0]
        """ plt.plot(y_train_one_day, label='True Values')
        plt.plot(train_predict_one_day, label='Predictions')
        plt.legend()
        plt.title('Training Data Predictions')
        plt.show() """
        # Creating traces for true values and predictions
        true_values_trace = go.Scatter(
            x=list(range(len(y_train_one_day))),
            y=y_train_one_day,
            mode='lines',
            name='True Values',
            line=dict(color='blue')
        )

        predictions_trace = go.Scatter(
            x=list(range(len(train_predict_one_day))),
            y=train_predict_one_day,
            mode='lines',
            name='Predictions',
            line=dict(color='red')
        )

        # Defining layout
        layout = go.Layout(
            title='Training Data Predictions',
            xaxis=dict(title='Days'),
            yaxis=dict(title='Close Prices')
        )

        # Creating the figure
        fig2 = go.Figure(data=[true_values_trace, predictions_trace], layout=layout)

        # Generating the plot
        plot_div_2= plot(fig2, output_type='div')

        # Plot testing predictions
        y_test_one_day = y_test[:, 0]
        test_predict_one_day = test_predict[:, 0]
        """ plt.plot(y_test_one_day, label='True Values')
        plt.plot(test_predict_one_day, label='Predictions')
        plt.legend()
        plt.title('Testing Data Predictions')
        plt.show() """
        # Creating traces for true values and predictions
        true_values_trace = go.Scatter(
            x=list(range(len(y_test_one_day))),
            y=y_test_one_day,
            mode='lines',
            name='True Values',
            line=dict(color='blue')
        )

        predictions_trace = go.Scatter(
            x=list(range(len(test_predict_one_day))),
            y=test_predict_one_day,
            mode='lines',
            name='Predictions',
            line=dict(color='red')
        )

        # Defining layout
        layout = go.Layout(
            title='Testing Data Predictions',
            xaxis=dict(title='Days'),
            yaxis=dict(title='Closed Prices')
        )

        # Creating the figure
        fig3 = go.Figure(data=[true_values_trace, predictions_trace], layout=layout)

        # Generating the plot
        plot_div_3 = plot(fig3, output_type='div')

        # Calculate RMSE, MAE, and R2 for training data
        train_rmse = np.sqrt(mean_squared_error(y_train, train_predict))
        train_r2 = r2_score(y_train, train_predict)
        train_mae = mean_absolute_error(y_train, train_predict)
        print("Training MAE:", train_mae)
        print("Training RMSE:", train_rmse)
        print("Training R2:", train_r2)

        # Calculate RMSE, MAE, and R2 for testing data
        test_rmse = np.sqrt(mean_squared_error(y_test, test_predict))
        test_r2 = r2_score(y_test, test_predict)
        test_mae = mean_absolute_error(y_test, test_predict)
        print("Testing MAE:", test_mae)
        print("Testing RMSE:", test_rmse)
        print("Testing R2:", test_r2)

        # Predict the next 7 days
        last_week_data = scaled_data[-time_steps:, :]
        last_week_data = np.reshape(last_week_data, (1, time_steps, 1))
        predictions = model.predict(last_week_data)
        predictions = scaler.inverse_transform(predictions)
        predicted_close_prices = predictions[0]

        # Generate forecast dates
        last_date = df['Date'].iloc[-1]
        forecast_dates = pd.date_range(start=last_date + pd.DateOffset(days=1), periods=prediction_days, freq='D')
        df_predictions = pd.DataFrame({'date': forecast_dates, 'close_price': predicted_close_prices.flatten()})
        print(df_predictions)

        """ # Generate Plotly chart
        chart_data = go.Scatter(x=forecast_dates, y=predicted_close_prices, mode='lines', name='Predicted Close Prices')
        layout = go.Layout(title='Predicted Close Prices Over Time', xaxis=dict(title='Date'), yaxis=dict(title='Close Price'))
        fig = go.Figure(data=[chart_data], layout=layout)
        plot_div = plot(fig, output_type='div') """

        """ # Create Plotly graph for the last `time_steps` days of historical data and the forecast
        historical_dates = df['Date'].iloc[-(time_steps + prediction_days):]
        historical_prices = df['Close'].iloc[-(time_steps + prediction_days):]

        historical_trace = go.Scatter(x=historical_dates, y=historical_prices, mode='lines', name='Historical Close Prices', line=dict(color='blue'))
        forecast_trace = go.Scatter(x=forecast_dates, y=predicted_close_prices, mode='lines', name='Predicted Close Prices', line=dict(color='red'))

        layout = go.Layout(title='Historical and Predicted Close Prices', xaxis=dict(title='Date'), yaxis=dict(title='Close Price'))
        fig = go.Figure(data=[historical_trace, forecast_trace], layout=layout)
        plot_div = plot(fig, output_type='div') """

        # Create Plotly graph for all historical data and the forecast
        historical_dates = df['Date']
        historical_prices = df['Close']

        historical_trace = go.Scatter(
            x=historical_dates,
            y=historical_prices,
            mode='lines',
            name='Historical Close Prices',
            line=dict(color='blue')
        )

        forecast_trace = go.Scatter(
            x=forecast_dates,
            y=predicted_close_prices,
            mode='lines',
            name='Predicted Close Prices',
            line=dict(color='red')
        )

        layout = go.Layout(
            title='Historical and Predicted Close Prices',
            xaxis=dict(title='Date'),
            yaxis=dict(title='Close Price')
        )

        fig = go.Figure(data=[historical_trace, forecast_trace], layout=layout)
        plot_div = plot(fig, output_type='div')

        # Extracting data
        epochs = list(range(1, len(history.history['loss']) + 1))
        training_loss = history.history['loss']
        validation_loss = history.history['val_loss']

        # Creating traces for training and validation loss
        training_trace = go.Scatter(
            x=epochs,
            y=training_loss,
            mode='lines',
            name='Training Loss',
            line=dict(color='blue')
        )

        validation_trace = go.Scatter(
            x=epochs,
            y=validation_loss,
            mode='lines',
            name='Validation Loss',
            line=dict(color='red')
        )

        # Defining layout
        layout = go.Layout(
            title='Training and Validation Loss',
            xaxis=dict(title='Epoch'),
            yaxis=dict(title='Loss')
        )

        # Creating the figure
        fig1= go.Figure(data=[training_trace, validation_trace], layout=layout)

        # Generating the plot
        plot_div_1 = plot(fig1, output_type='div')

        return df_predictions, train_rmse, test_rmse, train_r2, test_r2,train_mae,test_mae,plot_div,plot_div_1,plot_div_2,plot_div_3
        #return(df_predictions)

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# Example usage
#df_predictions, train_rmse, test_rmse, train_r2, test_r2 = lstm_model('nbl.csv')
