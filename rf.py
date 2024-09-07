

def random_forest_model(company,prediction_days):
    import pandas as pd
    import numpy as np
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    import matplotlib.pyplot as plt
    import plotly.graph_objs as go
    from plotly.offline import plot
    
    try:
        prediction_days=prediction_days
        time_steps = prediction_days
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
        X_train, y_train = [], []
        for i in range(len(train_data) - time_steps - prediction_days):
            X_train.append(train_data[i:i + time_steps, 0])
            y_train.append(train_data[i + time_steps:i + time_steps + prediction_days, 0])
        X_train, y_train = np.array(X_train), np.array(y_train)

        # Prepare the testing data
        X_test, y_test = [], []
        for i in range(len(test_data) - time_steps - prediction_days):
            X_test.append(test_data[i:i + time_steps, 0])
            y_test.append(test_data[i + time_steps:i + time_steps + prediction_days, 0])
        X_test, y_test = np.array(X_test), np.array(y_test)

        # Reshape the data for Random Forest (samples, time_steps)
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1]))
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1]))

        # Define and train the Random Forest model
        model = RandomForestRegressor(n_estimators=100, random_state=16)
        model.fit(X_train, y_train)

        # Predict the next 7 days
        last_week_data = scaled_data[-time_steps:, :]
        last_week_data = np.reshape(last_week_data, (1, time_steps))
        predictions = model.predict(last_week_data)
        predictions = scaler.inverse_transform(predictions)
        
        forecast_dates = pd.date_range(start=df['Date'].iloc[-1] + pd.DateOffset(days=1), periods=prediction_days, freq='D')
        df_predictions = pd.DataFrame({'date': forecast_dates, 'close_price': predictions.flatten()})
        print(df_predictions)

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
            x=df_predictions['date'],
            y=df_predictions['close_price'],  # Use predictions instead of predicted_close_prices
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
       
        # Make predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Inverse transform the predictions
        train_predict = scaler.inverse_transform(train_predict)
        test_predict = scaler.inverse_transform(test_predict)
        y_train = scaler.inverse_transform(y_train)
        y_test = scaler.inverse_transform(y_test)

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

        # Plot training predictions
        """ plt.plot(y_train[:, 0], label='True Values')
        plt.plot(train_predict[:, 0], label='Predictions')
        plt.legend()
        plt.title('Training Data Predictions')
        plt.show() """
        y_train_one_day = y_train[:, 0]
        train_predict_one_day = train_predict[:, 0]
        
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
        """ plt.plot(y_test[:, 0], label='True Values')
        plt.plot(test_predict[:, 0], label='Predictions')
        plt.legend()
        plt.title('Testing Data Predictions')
        plt.show() """
        y_test_one_day = y_test[:, 0]
        test_predict_one_day = test_predict[:, 0]
        
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
            yaxis=dict(title='Close Prices')
        )

        # Creating the figure
        fig3 = go.Figure(data=[true_values_trace, predictions_trace], layout=layout)

        # Generating the plot
        plot_div_3 = plot(fig3, output_type='div')

        # Plot forecasted values
        """ plt.plot(df['Date'], df['Close'], label='True Values')
        plt.plot(df_predictions['date'], df_predictions['close_price'], label='Forecasted Values')
        plt.legend()
        plt.title('Forecasted Values')
        plt.show() """

        

        return df_predictions, train_rmse, test_rmse, train_r2, test_r2,train_mae,test_mae,plot_div,None,plot_div_2,plot_div_3

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# Example usage
#df_predictions_rf, train_rmse_rf, test_rmse_rf, train_r2_rf, test_r2_rf = random_forest_model('nbl.csv')
