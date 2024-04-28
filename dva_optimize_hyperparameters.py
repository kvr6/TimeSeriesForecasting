import logging
import pandas as pd
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from config import CONFIG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def optimize_hyperparameters():
    try:
        # Load your data
        df = pd.read_csv(CONFIG['data_path'])
        df['ds'] = pd.to_datetime(df['ds'])
        df['y'] = pd.to_numeric(df['y'])

        # Define the search space for hyperparameters
        space = {
            'changepoint_prior_scale': hp.loguniform('changepoint_prior_scale', -5, 0),
            'seasonality_prior_scale': hp.loguniform('seasonality_prior_scale', -5, 0),
            'fourier_order': hp.choice('fourier_order', [5, 10, 15])
        }

        # Define the objective function to minimize
        def objective(params):
            m = Prophet(
                changepoint_prior_scale=params['changepoint_prior_scale'],
                seasonality_prior_scale=params['seasonality_prior_scale']
            )
            m.add_seasonality(name='quarterly', period=91.25, fourier_order=int(params['fourier_order']))
            m.fit(df)
            df_cv = cross_validation(m, initial='730 days', period='180 days', horizon='365 days')
            df_p = performance_metrics(df_cv)
            # Use MAPE as the loss
            mape = df_p['mape'].mean()
            return {'loss': mape, 'status': STATUS_OK}

        # Perform optimization
        trials = Trials()
        best_params = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=100, trials=trials)

        logger.info(f"Best parameters found: {best_params}")

        # Convert `fourier_order` from index to actual value
        best_params['fourier_order'] = [5, 10, 15][best_params['fourier_order']]

        # Save the best parameters to a CSV file
        pd.DataFrame([best_params]).to_csv(CONFIG['best_params_path'], index=False)

    except Exception as e:
        logger.exception(f"An error occurred during hyperparameter optimization: {str(e)}")
        raise

if __name__ == '__main__':
    optimize_hyperparameters()

