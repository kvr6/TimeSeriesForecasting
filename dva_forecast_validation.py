import logging
import pandas as pd
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway
from config import CONFIG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def compute_mape(actual, forecast):
    """
    Compute Mean Absolute Percentage Error (MAPE) between actual and forecast values.
    """
    actual, forecast = pd.Series(actual), pd.Series(forecast)
    # Avoid division by zero and drop missing values
    mask = actual != 0
    return ((forecast[mask] - actual[mask]).abs() / actual[mask]).mean() * 100

def validate_forecasts():
    try:
        forecast_df = pd.read_csv(CONFIG['forecast_output_path'])
        actual_df = pd.read_csv(CONFIG['actual_values_path'])
        forecast_periods = [col for col in forecast_df.columns if 'fcst_' in col]
        actual_periods = [col.replace('fcst_', 'actual_') for col in forecast_periods]

        # Create a Prometheus CollectorRegistry
        registry = CollectorRegistry()

        # Create Prometheus gauges for MAPE scores
        mape_gauges = {}
        for fcst_col in forecast_periods:
            mape_gauges[fcst_col] = Gauge(f'mape_{fcst_col}', f'MAPE for {fcst_col}', registry=registry)

        # Compute MAPE for each forecast period and set the corresponding gauge
        mape_scores = {}
        for fcst_col, act_col in zip(forecast_periods, actual_periods):
            mape = compute_mape(actual_df[act_col], forecast_df[fcst_col])
            mape_scores[fcst_col] = mape
            mape_gauges[fcst_col].set(mape)
            logger.info(f"MAPE for {fcst_col}: {mape:.2f}%")

        # Push the metrics to the Prometheus Pushgateway
        push_to_gateway(CONFIG['prometheus_gateway'], job='dva_forecast_validation', registry=registry)

        pd.DataFrame([mape_scores]).to_csv(CONFIG['mape_scores_path'], index=False)
        logger.info(f"MAPE scores saved to {CONFIG['mape_scores_path']}")

        return mape_scores

    except Exception as e:
        logger.exception(f"An error occurred during forecast validation: {str(e)}")
        raise

if __name__ == '__main__':
    validate_forecasts()


