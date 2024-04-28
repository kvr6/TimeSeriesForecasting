import logging
import pandas as pd
from prophet import Prophet
import statsmodels.api as sm
from statsmodels.formula.api import ols
from config import CONFIG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_prime_day_significance(data, events):
    """Performs regression to test the significance of Prime Day and returns if it's significant."""
    try:
        event_date = events['EventDate'].min()  # Considering the first event date

        # Define the window around the event
        pre_event_window = data[data.index < event_date]
        during_event_window = data[data.index == event_date]
        post_event_window = data[data.index > event_date]

        # Create a combined dataset for regression with dummy variables
        data['PreEvent'] = 0
        data.loc[data.index < event_date, 'PreEvent'] = 1
        data['DuringEvent'] = 0
        data.loc[data.index == event_date, 'DuringEvent'] = 1
        data['PostEvent'] = 0
        data.loc[data.index > event_date, 'PostEvent'] = 1

        # Fit a regression model
        model = ols('Revenue ~ PreEvent + DuringEvent + PostEvent', data=data).fit()
        logger.info(model.summary())

        # Return true if the event coefficient is significant
        return model.pvalues['DuringEvent'] < 0.05

    except Exception as e:
        logger.exception(f"An error occurred during Prime Day significance check: {str(e)}")
        raise

def generate_forecasts():
    try:
        # Load ad revenue data and set date index
        data = pd.read_csv(CONFIG['ad_revenue_data_path'])
        data['Date'] = pd.to_datetime(data['Date'])
        data.set_index('Date', inplace=True)
        data['Revenue'] = pd.to_numeric(data['Revenue'])

        # Load Prime Day dates from CSV
        prime_days = pd.read_csv(CONFIG['prime_days_path'])
        prime_days['EventDate'] = pd.to_datetime(prime_days['EventDate'])

        # Check if Prime Day significantly affects ad revenue
        significant = check_prime_day_significance(data, prime_days)

        # Configure holidays in Prophet if significant
        holidays = pd.DataFrame({
            'holiday': 'PrimeDay',
            'ds': prime_days['EventDate'],
            'lower_window': 0,
            'upper_window': 1,
        }) if significant else None

        # Load optimized hyperparameters
        hyperparameters = pd.read_csv(CONFIG['best_params_path']).iloc[0]
        model = Prophet(
            changepoint_prior_scale=hyperparameters['changepoint_prior_scale'],
            seasonality_prior_scale=hyperparameters['seasonality_prior_scale'],
            holidays=holidays
        )

        # Fit and forecast
        model.add_seasonality(name='quarterly', period=91.25, fourier_order=5)
        model.fit(data.reset_index().rename(columns={'Date': 'ds', 'Revenue': 'y'}))
        future = model.make_future_dataframe(periods=180)
        forecast = model.predict(future)

        # Save forecast to CSV
        forecast[['ds', 'yhat']].to_csv(CONFIG['forecast_output_path'], index=False)

        # Post-processing to handle negatives and product exclusions
        final_data = pd.read_csv(CONFIG['forecast_output_path'])
        final_data.loc[final_data['yhat'] < 1.1, 'yhat'] = 1.1  # Set revenue floor to 1.1
        products_to_remove = pd.read_csv(CONFIG['products_to_remove_path'])['ad_product'].tolist()
        final_data = final_data[~final_data['product_id'].isin(products_to_remove)]
        final_data.to_csv(CONFIG['final_forecast_path'], index=False)

    except Exception as e:
        logger.exception(f"An error occurred during forecast generation: {str(e)}")
        raise

if __name__ == '__main__':
    generate_forecasts()

