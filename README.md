# TimeSeriesForecasting
Repository to share time series forecasting work using Meta's Prophet library.  

### High-level overview of the codes and sample productionalization files:
- dva_optimize_hyperparameters.py: Script for tuning Prophet parameters refactored to include error handling and logging.
- dva_forecast_generation.py: Script for generating forecasts improved error handling and logging.
- dva_forecast_validation.py: Script for validating generated forecasts (MAPE values). Added Prometheus specific code to enable logging/monitoring of MAPE scores.
- Dockerfile: Sets up a Python environment, installs required dependencies, copies the application code into the container, and specifies the default command to run the forecast generation script.
- kubernetes_deploy: Creates Kubernetes manifests for deploying the application.
- CronJob: Schedule daily forecast generation

### For visualizing via Grafana, a sample production setup could be: 
- Deplpoy Grafana using Kubernetes manifests
- Configure Grafana to connect to the data source (e.g. Prometheus) where model performance metrics are stored
- Create Grafana dashboards to visualize the model performance metrics (e.g. MAPE) over time. 
