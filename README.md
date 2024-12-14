# Project
## Predicting Ride Tips with NYC TLC Data

### Data Processing Infrastructure
- Cloud Platform: The entire data processing pipeline is built and executed on Google Cloud Platform (GCP), leveraging its scalability and efficiency.
- Virtual Machine (VM): A Google Cloud Virtual Machine (VM) environment is used to run all data processing tasks, ensuring flexibility and computational power for handling large datasets.
- Dataproc: Google Cloud Dataproc was utilized for distributed data processing and scalable analysis. Dataproc provides a managed Apache Hadoop and Apache Spark service, enabling efficient TLC trip data batch processing.

### Description
- This project analyzed tipping behavior in NYC for-hire vehicles using high-volume trip record data from the NYC Taxi & Limousine Commission (TLC). The dataset covers trips from January 2022 to August 2023, including such as pickup/drop-off times, locations, distances, and fares.

### Proposal
- By analyzing key trip characteristics, this project aims to uncover patterns in tipping behavior and predict both the likelihood and amount of tips. These insights can help drivers identify factors influencing tips, enabling them to optimize their earnings by focusing on higher-tipping routes, times, or conditions. Additionally, companies can leverage these findings to enhance customer service, improve driver incentive programs, and refine operational strategies, ultimately boosting both driver satisfaction and customer loyalty.

## Data Acquistion
- I first created a bucket. In the bucket, I created folders for landing, cleaned, trusted, code and models
Created a new VM instance
Opened the terminal on GCP VM, I first created a shell script for downloading data:
touch download_tlc_data.sh 
Open the script: 
nano download_tlc_data.sh 
Typed in my script code as below (copy the links for parquet files from the TLC website and save them with appropriate names):

```ruby
# Define the URLs for each month's Parquet file from Jan 2022 to Aug 2023
declare -A data_files=(
    ["fhv_trip_records_2022_01.parquet"]="https://d37ci6vzurychx.cloudfront.net/trip-data/fhvhv_tripdata_2022-01.parquet"
    ["fhv_trip_records_2022_02.parquet"]="https://d37ci6vzurychx.cloudfront.net/trip-data/fhvhv_tripdata_2022-02.parquet"
    ["fhv_trip_records_2022_03.parquet"]="https://d37ci6vzurychx.cloudfront.net/trip-data/fhvhv_tripdata_2022-03.parquet"
    ["fhv_trip_records_2022_04.parquet"]="https://d37ci6vzurychx.cloudfront.net/trip-data/fhvhv_tripdata_2022-04.parquet"
    ["fhv_trip_records_2022_05.parquet"]="https://d37ci6vzurychx.cloudfront.net/trip-data/fhvhv_tripdata_2022-05.parquet"
    ["fhv_trip_records_2022_06.parquet"]="https://d37ci6vzurychx.cloudfront.net/trip-data/fhvhv_tripdata_2022-06.parquet"
    ["fhv_trip_records_2022_07.parquet"]="https://d37ci6vzurychx.cloudfront.net/trip-data/fhvhv_tripdata_2022-07.parquet"
    ["fhv_trip_records_2022_08.parquet"]="https://d37ci6vzurychx.cloudfront.net/trip-data/fhvhv_tripdata_2022-08.parquet"
    ["fhv_trip_records_2022_09.parquet"]="https://d37ci6vzurychx.cloudfront.net/trip-data/fhvhv_tripdata_2022-09.parquet"
    ["fhv_trip_records_2022_10.parquet"]="https://d37ci6vzurychx.cloudfront.net/trip-data/fhvhv_tripdata_2022-10.parquet"
    ["fhv_trip_records_2022_11.parquet"]="https://d37ci6vzurychx.cloudfront.net/trip-data/fhvhv_tripdata_2022-11.parquet"
    ["fhv_trip_records_2022_12.parquet"]="https://d37ci6vzurychx.cloudfront.net/trip-data/fhvhv_tripdata_2022-12.parquet"
    ["fhv_trip_records_2023_01.parquet"]="https://d37ci6vzurychx.cloudfront.net/trip-data/fhvhv_tripdata_2023-01.parquet"
    ["fhv_trip_records_2023_02.parquet"]="https://d37ci6vzurychx.cloudfront.net/trip-data/fhvhv_tripdata_2023-02.parquet"
    ["fhv_trip_records_2023_03.parquet"]="https://d37ci6vzurychx.cloudfront.net/trip-data/fhvhv_tripdata_2023-03.parquet"
    ["fhv_trip_records_2023_04.parquet"]="https://d37ci6vzurychx.cloudfront.net/trip-data/fhvhv_tripdata_2023-04.parquet"
    ["fhv_trip_records_2023_05.parquet"]="https://d37ci6vzurychx.cloudfront.net/trip-data/fhvhv_tripdata_2023-05.parquet"
    ["fhv_trip_records_2023_06.parquet"]="https://d37ci6vzurychx.cloudfront.net/trip-data/fhvhv_tripdata_2023-06.parquet"
    ["fhv_trip_records_2023_07.parquet"]="https://d37ci6vzurychx.cloudfront.net/trip-data/fhvhv_tripdata_2023-07.parquet"
    ["fhv_trip_records_2023_08.parquet"]="https://d37ci6vzurychx.cloudfront.net/trip-data/fhvhv_tripdata_2023-08.parquet"
)

# Download and upload each file to GCS
for file in "${!data_files[@]}"; do
    url="${data_files[$file]}"
    echo "Downloading $file from $url"
    curl -o "$file" "$url" # Downloads the Parquet file
    echo "Uploading $file to GCS"
    gsutil cp "$file" "$GCS_BUCKET/$file" # Uploads the file to my GCS bucket
done
```

- Saved and exited after completing the code (pressed control x, y, then hit enter). 
- Lastly, set permissions and run the script:
```
chmod +x download_tlc_data.sh
./download_tlc_data.sh
```


## EDA
```ruby
!pip install pyarrow fastparquet

# Import required libraries
import pyarrow
import fastparquet
import pandas as pd
import seaborn as sns
from tabulate import tabulate
import matplotlib.pyplot as plt

# Repeat the following codes for the rest of the files (I initially tried looping through all files, but they were too large to handle together, so I ended up processing them one at a time)
# The file location
fhv_trip2201 = 'gs://my-bigdata-project-xq/landing/fhv_trip_records_2022_01.parquet'

# Read the Parquet file
trips_df2201 = pd.read_parquet(fhv_trip2201, engine='pyarrow')

def perform_EDA(df: pd.DataFrame, filename: str):
    """
    Perform EDA on a DataFrame and output structured information for readability.
    """
    print(f"\n--- {filename} ---")
    duplicates = trips_df2201.duplicated().sum()
    print("\nNumber of Duplicates:", duplicates)
    
    print(f"Info:")
    print(df.info())
    
    # Format and display `describe` output
    print("\nDescribe Summary:")
    describe_df = df.describe().transpose().reset_index()
    print(tabulate(describe_df, headers="keys", tablefmt="fancy_grid", floatfmt=".2f"))

    # Columns with null values
    null_counts = df.isnull().sum()
    print("\nColumns with Null Values:")
    print(null_counts[null_counts > 0].to_string())

    # Rows with any null values
    rows_with_null_values = df.isnull().any(axis=1).sum()
    print(f"\nNumber of Rows with Null Values: {rows_with_null_values}")

# Perform EDA on the file
perform_EDA(trips_df2201, fhv_trip2201)

# Box Plot of trip time
plt.figure(figsize=(10, 6))
sns.boxplot(x=trips_df2201['trip_time'])
plt.title('2022-01 Box Plot of trip time')
plt.xlabel('trip time')
plt.show()

# Box Plot of base passenger fare
plt.figure(figsize=(10, 6))
sns.boxplot(x=trips_df2201['base_passenger_fare'])
plt.title('2022-01 Box Plot of Base Passenger Fare')
plt.xlabel('Base Passenger Fare')
plt.show()

# Box Plot of tolls
plt.figure(figsize=(10, 6))
sns.boxplot(x=trips_df2201['tolls'])
plt.title('2022-01 Box Plot of Tolls')
plt.xlabel('Tolls')
plt.show()

# Box Plot of tips
plt.figure(figsize=(10, 6))
sns.boxplot(x=trips_df2201['tips'])
plt.title('2022-01 Box Plot of tips')
plt.xlabel('tips')
plt.show()

# Box Plot of driver pay
plt.figure(figsize=(10, 6))
sns.boxplot(x=trips_df2201['driver_pay'])
plt.title('2022-01 Box Plot of driver pay')
plt.xlabel('driver pay')
plt.show()
```


## Data Cleaning
- merge taxi zone information (match the information with ‘PULocationID’ and ‘DOLocationID’); drop 'hvfhs_license_num', 'dispatching_base_num' and extra LocationID column ('LocationID_pulocation', 'LocationID_dolocation_lookup')

```ruby
# Import necessary libraries
import pandas as pd
import gcsfs
import os

# Initialize GCS file system
fs = gcsfs.GCSFileSystem()

# Define GCS bucket paths
landing_folder = 'gs://my-bigdata-project-xq/landing/'
cleaned_folder = 'gs://my-bigdata-project-xq/cleaned/'
taxi_zone_lookup_folder = 'gs://my-bigdata-project-xq/taxi_zone_lookup/'

# Define the data cleaning function
def clean_data(file_path):
    # Load data
    with fs.open(file_path) as f:
        df = pd.read_parquet(f)
    
    # DEBUG: Check the first few rows of the raw file
    print(f"Loaded file: {file_path}")
    print(df.head())

    # Locate the taxi zone lookup CSV
    taxi_zone_lookup_files = fs.glob(f"{taxi_zone_lookup_folder}*.csv")
    if not taxi_zone_lookup_files:
        raise FileNotFoundError("No taxi zone lookup CSV found in the specified folder.")
    
    taxi_zone_lookup_path = taxi_zone_lookup_files[0]
    print(f"Loading Taxi Zone Lookup from: {taxi_zone_lookup_path}")
    
    # Load Taxi Zone lookup CSV
    with fs.open(taxi_zone_lookup_path) as tz:
        taxi_zone_df = pd.read_csv(tz)

    # DEBUG: Check the first few rows of the taxi zone file
    print("Taxi Zone Lookup Data:")
    print(taxi_zone_df.head())

    # Merge the taxi zone information with PULocationID
    df = df.merge(taxi_zone_df[['LocationID', 'Borough', 'Zone', 'service_zone']], 
                  how='left', 
                  left_on='PULocationID', 
                  right_on='LocationID', 
                  suffixes=('_pulocation', '_pulocation_lookup'))
    
    # Merge the taxi zone information with DOLocationID
    df = df.merge(taxi_zone_df[['LocationID', 'Borough', 'Zone', 'service_zone']], 
                  how='left', 
                  left_on='DOLocationID', 
                  right_on='LocationID', 
                  suffixes=('_pulocation', '_dolocation_lookup'))
    
    # Drop the extra LocationID column
    df = df.drop(columns=['LocationID_pulocation', 'LocationID_dolocation_lookup'])

    # Data Cleaning Steps
    df['is_duplicate'] = df.duplicated(keep=False)
    if 'PULocationID' in df.columns and 'DOLocationID' in df.columns:
        df['PULocationID'] = df['PULocationID'].astype('Int32')
        df['DOLocationID'] = df['DOLocationID'].astype('Int32')
    
    columns_to_drop = ['originating_base_num', 'on_scene_datetime', 'hvfhs_license_num', 'dispatching_base_num']
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
    df = df[(df['base_passenger_fare'] >= 0) & (df['driver_pay'] >= 0)]
    df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]
    
    outlier_columns = ['trip_time', 'base_passenger_fare', 'tolls', 'tips',
                       'driver_pay', 'trip_miles', 'sales_tax', 'congestion_surcharge']
    threshold = 3
    for col in outlier_columns:
        if col in df.columns:
            col_mean = df[col].mean()
            col_std = df[col].std()
            df = df[(df[col] - col_mean).abs() <= threshold * col_std]
    
    # Save cleaned data to cleaned folder in Parquet format
    cleaned_file_path = os.path.join(cleaned_folder, os.path.basename(file_path))
    with fs.open(cleaned_file_path, 'wb') as f:
        df.to_parquet(f)
    print(f"Cleaned file saved: {cleaned_file_path}")

# Use glob to match all Parquet files in the landing folder
files_to_clean = fs.glob(f"{landing_folder}*.parquet")  # Matches all .parquet files

# DEBUG: Print matched files
print("Files to clean:")
print(files_to_clean)

# Loop through and clean each file
for file_path in files_to_clean:
    clean_data(file_path)
```


## Feature Engineering and Modeling
```ruby
from pyspark.sql.functions import *
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql import functions as F
spark.conf.set("spark.sql.debug.maxToStringFields", "200")

sdf = spark.read.parquet("gs://my-bigdata-project-xq/cleaned/")
sdf.printSchema()

# List of US holidays for the years 2022 and 2023
us_holidays = [
    '2022-01-01', '2022-01-17', '2022-02-21', '2022-05-30', '2022-06-19', '2022-07-04', '2022-09-05', '2022-10-10', 
    '2022-11-11', '2022-11-24', '2022-12-25',  # Common holidays in 2022
    '2023-01-01', '2023-01-16', '2023-02-20', '2023-05-29', '2023-06-19', '2023-07-04',  # Common holidays in 2023 till August
]

# Engineer additional date feature columns based on the pickup_datetime
sdf = sdf.withColumn("pickup_year", F.year(F.col("pickup_datetime")))
sdf = sdf.withColumn("pickup_month", F.month(F.col("pickup_datetime")))   # Numeric month like 11
sdf = sdf.withColumn("pickup_yearmonth", F.date_format(F.col("pickup_datetime"), "yyyy-MM"))   # Like 2023-01   2023-02 etc.
sdf = sdf.withColumn("pickup_dayofweek", F.date_format(F.col("pickup_datetime"), "EEEE"))         # 'Monday' 'Tuesday' etc.
sdf = sdf.withColumn("pickup_weekend", F.when(F.col("pickup_dayofweek").isin('Saturday', 'Sunday'), 1).otherwise(0))

# Check if the pickup_datetime falls on a holiday
sdf = sdf.withColumn("pickup_is_holiday", F.when(F.col("pickup_datetime").cast("date").isin(*us_holidays), 1).otherwise(0))

# Define time of day categories based on the hour
sdf = sdf.withColumn("pickup_time_of_day", 
                     F.when((F.hour(F.col("pickup_datetime")) >= 5) & (F.hour(F.col("pickup_datetime")) < 12), "Morning")
                      .when((F.hour(F.col("pickup_datetime")) >= 12) & (F.hour(F.col("pickup_datetime")) < 17), "Afternoon")
                      .when((F.hour(F.col("pickup_datetime")) >= 17) & (F.hour(F.col("pickup_datetime")) < 21), "Evening")
                      .otherwise("Night"))  # Night from 9 PM to 5 AM

# Drop rows with any null values
cleaned_sdf = sdf.dropna(how="any")

# Split the entire cleaned dataset into training and test sets
trainingData, testData = cleaned_sdf.randomSplit([0.70, 0.30], seed=42)

# Step 1: Create StringIndexers for categorical variables
indexer = StringIndexer(
    inputCols=[
        "borough_pulocation", "borough_dolocation_lookup", 
        "zone_pulocation", "service_zone_pulocation",
        "zone_dolocation_lookup", "service_zone_dolocation_lookup",
        "pickup_year", "pickup_month", "pickup_yearmonth", "pickup_dayofweek",
        "pickup_weekend", "pickup_is_holiday", "pickup_time_of_day"
    ],
    outputCols=[
        "borough_pulocationIndex", "borough_dolocation_lookupIndex", 
        "zone_pulocationIndex", "service_zone_pulocationIndex",
        "zone_dolocation_lookupIndex", "service_zone_dolocation_lookupIndex",
        "pickup_yearIndex", "pickup_monthIndex", "pickup_yearmonthIndex", "pickup_dayofweekIndex",
        "pickup_weekendIndex", "pickup_is_holidayIndex", "pickup_time_of_dayIndex"
    ], 
    handleInvalid="keep"
)

# Step 2: Create OneHotEncoders for categorical variables
encoder = OneHotEncoder(
    inputCols=[
        "borough_pulocationIndex", "borough_dolocation_lookupIndex", 
        "zone_pulocationIndex", "service_zone_pulocationIndex",
        "zone_dolocation_lookupIndex", "service_zone_dolocation_lookupIndex",
        "pickup_yearIndex", "pickup_monthIndex", "pickup_yearmonthIndex", "pickup_dayofweekIndex",
        "pickup_weekendIndex", "pickup_is_holidayIndex", "pickup_time_of_dayIndex"
    ],
    outputCols=[
        "borough_pulocationVector", "borough_dolocation_lookupVector", 
        "zone_pulocationVector", "service_zone_pulocationVector",
        "zone_dolocation_lookupVector", "service_zone_dolocation_lookupVector",
        "pickup_yearVector", "pickup_monthVector", "pickup_yearmonthVector", "pickup_dayofweekVector",
        "pickup_weekendVector", "pickup_is_holidayVector", "pickup_time_of_dayVector"
    ],
    dropLast=True,
    handleInvalid="keep"
)

# Step 3: Assemble all features into a single vector
assembler = VectorAssembler(
    inputCols=[
        "borough_pulocationVector", "borough_dolocation_lookupVector", 
        "zone_pulocationVector", "service_zone_pulocationVector",
        "zone_dolocation_lookupVector", "service_zone_dolocation_lookupVector",
        "pickup_yearVector", "pickup_monthVector", "pickup_yearmonthVector", "pickup_dayofweekVector",
        "pickup_weekendVector", "pickup_is_holidayVector", "pickup_time_of_dayVector",
        'trip_miles', 'base_passenger_fare', 'tolls', 'driver_pay', 'trip_time', "congestion_surcharge"
    ],
    outputCol="features"
)

# Step 4: Define the model pipeline
linear_reg = LinearRegression(labelCol='tips')
pipeline = Pipeline(stages=[indexer, encoder, assembler, linear_reg])

# Hyperparameter tuning using CrossValidator
paramGrid = ParamGridBuilder().build()
crossval = CrossValidator(
    estimator=pipeline, 
    estimatorParamMaps=paramGrid, 
    evaluator=RegressionEvaluator(labelCol='tips'), 
    numFolds=3
)

# Train the model using cross-validation
cvModel = crossval.fit(trainingData)

# Get the best model and evaluate performance
bestModel = cvModel.bestModel

# Show the average performance over the three folds
print(f"Average metric {cvModel.avgMetrics}")

# Make predictions on the test set
predictions = bestModel.transform(testData)

# Show predictions and evaluate metrics
predictions.select(
    "borough_pulocation", "borough_dolocation_lookup", 
    "zone_pulocation", "service_zone_pulocation", "zone_dolocation_lookup", "service_zone_dolocation_lookup",
    "pickup_datetime", "pickup_year", "pickup_month", "pickup_dayofweek", "pickup_weekend",
    "pickup_is_holiday", "pickup_time_of_day", 'trip_miles', 'base_passenger_fare', 'tolls', 
    'driver_pay', 'trip_time', 'tips', 'prediction'
).show(truncate=False)

# Evaluate RMSE and R2
evaluator = RegressionEvaluator(labelCol='tips')
rmse = evaluator.evaluate(predictions, {evaluator.metricName: 'rmse'})
r2 = evaluator.evaluate(predictions, {evaluator.metricName: 'r2'})

print(f"RMSE: {rmse}")
print(f"R2: {r2}")

# Save the best model to Google Cloud Storage
model_save_path = "gs://my-bigdata-project-xq/models/LR_trained_model"
cvModel.bestModel.write().overwrite().save(model_save_path)
print(f"Best model saved to {model_save_path}")

# Apply the pipeline to the training data to get processed data
processed_data = bestModel.transform(trainingData)

# Save the processed data (with features) to the /trusted folder
data_save_path = "gs://my-bigdata-project-xq/trusted/processed_data_with_features.parquet"
processed_data.write.mode("overwrite").parquet(data_save_path)
print(f"Processed data saved to {data_save_path}")
```


## Visualization
```ruby
import io
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Import some modules we will need later on
from pyspark.sql.functions import col, isnan, when, count, udf, to_date, year, month, date_format, size, split
from pyspark.ml.stat import Correlation
from pyspark.ml.feature import VectorAssembler
from google.cloud import storage

# Read my files
sdf = spark.read.parquet("gs://my-bigdata-project-xq/trusted/processed_data_with_features.parquet")

sdf.printSchema()

# Group by borough_pulocation and count orders on the entire dataset
df = (
    sdf.groupBy("borough_pulocation")
    .count()
    .sort("borough_pulocation")
    .toPandas()
)


# Create a bar plot for the order count by pick up borough
myplot = df.plot.bar('borough_pulocation', 'count', legend=False)
myplot.set(xlabel='Borough', ylabel='Order Count')
myplot.set(title='Order Count by Borough')
myplot.figure.set_tight_layout('tight')

# Save the plot to a local memory buffer
img_data = io.BytesIO()
plt.savefig(img_data, format='png', bbox_inches='tight')
img_data.seek(0)

# Connect to Google Cloud Storage
storage_client = storage.Client()
bucket = storage_client.get_bucket('my-bigdata-project-xq')
blob = bucket.blob("order_count_by_borough.png")

# Upload the image to GCP
blob.upload_from_file(img_data)

print("Visualization saved to Google Cloud Storage as 'order_count_by_borough.png'")

# Group by borough_dolocation and count orders
df = (
    sdf.groupBy("borough_dolocation_lookup")  # Group by drop-off location instead of pickup location
    .count()
    .sort("borough_dolocation_lookup")
    .toPandas()
)


# Create a bar plot for the order count by drop-off location
myplot = df.plot.bar('borough_dolocation_lookup', 'count', legend=False)
myplot.set(xlabel='Borough (Drop-off)', ylabel='Order Count')
myplot.set(title='Order Count by Drop-off Borough')
myplot.figure.set_tight_layout('tight')

# Save the plot to a local memory buffer
img_data = io.BytesIO()
plt.savefig(img_data, format='png', bbox_inches='tight')
img_data.seek(0)

# Connect to Google Cloud Storage
storage_client = storage.Client()
bucket = storage_client.get_bucket('my-bigdata-project-xq')
blob = bucket.blob("order_count_by_drop_off_borough.png")

# Upload the image to GCP
blob.upload_from_file(img_data)

print("Visualization saved to Google Cloud Storage as 'order_count_by_drop_off_borough.png'")


# Group by 'borough_pulocation' and calculate the mean tips for each location
location_tips = sdf.groupby('borough_pulocation').agg({'tips': 'mean'}).withColumnRenamed('avg(tips)', 'average_tips')

# Convert to Pandas for plotting
location_tips_pd = location_tips.toPandas()

# Create the bar plot for tips by location
sns_plot = location_tips_pd.plot(kind='bar', x='borough_pulocation', y='average_tips', figsize=(10, 6))

plt.title('Average Tips by Pickup Location')
plt.xlabel('Pickup Location')
plt.ylabel('Average Tip Amount')

# Save the plot to a local memory buffer
img_data = io.BytesIO()
sns_plot.figure.savefig(img_data, format='png', bbox_inches='tight')
img_data.seek(0)

# Connect to Google Cloud Storage
storage_client = storage.Client()
bucket = storage_client.get_bucket('my-bigdata-project-xq')
blob = bucket.blob("average_tips_by_pickup_location.png")

# Upload the image to GCP
blob.upload_from_file(img_data)

print("Visualization saved to Google Cloud Storage as 'average_tips_by_pickup_location.png'")

# Show the relationship between tips and prediction
sample_sdf = sdf.sample(withReplacement=False, fraction=0.01)
df = sample_sdf.select('tips', 'prediction').toPandas()

# Set the style for Seaborn plots
sns.set_style("white")

# Create a relationship plot between 'tips' and 'prediction'
sns_plot = sns.lmplot(x='tips', y='prediction', data=df, line_kws={"color": "red", "linestyle": "--"}, scatter_kws={"alpha": 1})

# Add plot titles and axis labels for clarity
sns_plot.set_axis_labels("Actual Tip Amount", "Predicted Tip Amount")
sns_plot.fig.suptitle("Actual vs. Predicted Tips")
sns_plot.fig.tight_layout()

# Save the plot to a local memory buffer
img_data = io.BytesIO()
sns_plot.savefig(img_data, format='png', bbox_inches='tight')
img_data.seek(0)

from google.cloud import storage
# Connect to Google Cloud Storage
storage_client = storage.Client()
bucket = storage_client.get_bucket('my-bigdata-project-xq')
blob = bucket.blob("actual_vs_predicted_tips.png")

# Upload the image to GCP
blob.upload_from_file(img_data)

print("Visualization saved to Google Cloud Storage as 'actual_vs_predicted_tips.png'")


# List of numeric columns for correlation matrix
numeric_columns = ['trip_miles', 'trip_time', 'base_passenger_fare', 'tolls', 
                   'bcf', 'sales_tax', 'congestion_surcharge', 'airport_fee', 
                   'tips', 'driver_pay']

# VectorAssembler to combine the numeric columns
vector_column = "correlation_features"
assembler = VectorAssembler(inputCols=numeric_columns, outputCol=vector_column)

# Transform the data
sdf_vector = assembler.transform(sdf).select(vector_column)

# Compute the correlation matrix
matrix = Correlation.corr(sdf_vector, vector_column).head()[0]  # Collect the correlation matrix
correlation_matrix = matrix.toArray()  # Convert to a numpy array

# Convert the correlation matrix to a Pandas DataFrame for better visualization
correlation_matrix_df = pd.DataFrame(data=correlation_matrix, columns=numeric_columns, index=numeric_columns)

# Set the style for Seaborn plots
sns.set_style("white")

# Create the heatmap for the correlation matrix
plt.figure(figsize=(16, 5))
hm = sns.heatmap(correlation_matrix_df, annot=True, cmap="Greens", 
                 xticklabels=correlation_matrix_df.columns.values, 
                 yticklabels=correlation_matrix_df.columns.values)

# Adjust layout and title
plt.title("Correlation Matrix of Numeric Features")
plt.tight_layout()

# Save the plot to a memory buffer
img_data = io.BytesIO()
plt.savefig(img_data, format='png', bbox_inches='tight')
img_data.seek(0)

# Connect to Google Cloud Storage
storage_client = storage.Client()
bucket = storage_client.get_bucket('my-bigdata-project-xq')  # Replace with your bucket name
blob = bucket.blob("correlation_matrix.png")

# Upload the image to GCP
blob.upload_from_file(img_data)

print("Visualization saved to Google Cloud Storage as 'correlation_matrix.png'")

# Show the plot
plt.show()
```


## Conclusion
- The model's performance is limited, as reflected by the low R² (0.0536) and RMSE (1.5324), suggesting that the features used do not adequately capture the tipping behavior. One potential issue is that the data is dominated by zero-dollar tips, making it difficult for the model to make meaningful predictions for non-zero tips. This imbalance may affect the model's ability to predict higher tips accurately.
- To improve the model, further feature engineering could be explored, along with different modeling approaches such as decision trees or random forests. Addressing the data imbalance (e.g., oversampling non-zero tips) and refining the data preprocessing steps (e.g., normalizing unrealistic low tip values) might also improve predictive accuracy.




