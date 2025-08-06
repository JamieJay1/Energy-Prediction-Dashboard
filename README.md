# Residential Energy Forecasting Dashboard

A smart, anomaly-aware LSTM-based forecasting dashboard for monitoring and predicting daily electricity consumption across residential buildings.

---

## Dataset

This project uses the **[Building Data Genome Project 2](https://www.kaggle.com/datasets/buds-lab/building-data-genome-project-2)**, which contains:
- Energy consumption data for 1,636 buildings
- Hourly readings across various meter types
- Weather data from NOAA
- Metadata on building location, use type, etc.

---

## Project Features

-  LSTM-based forecasting models with temporal & weather features
-  Anomaly detection using Isolation Forests
-  Error metrics: RMSE, MAE
-  Interactive dashboard (built with Plotly Dash)
-  Weather and anomaly overlays on prediction charts

---

## Setup Instructions

### Google Colab

1. Open the notebook in [Colab](https://colab.research.google.com/)
2. Mount your Google Drive:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
3.then Upload Your Kaggle.json file:
 ```python
   from google.colab import files
   files.upload()  # upload kaggle.json here
 ```

![Screenshot 2025-07-04 015930](https://github.com/user-attachments/assets/<img width="1395" height="500" alt="newplot (2)" src="https://github.com/user-attachments/assets/46f0b11a-46a2-4c57-9a6f-48f42dc7500c" />
2554c269-5616-4ac2-8b58-3e7cab795dc5)

<img width="1878" height="780" alt="Screenshot 2025-07-04 015930" src="https://github.com/user-attachments/assets/bf2042df-ff53-4c69-ad3c-7a9d0abfd330" />


