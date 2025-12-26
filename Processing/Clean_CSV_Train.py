#!/usr/bin/env python
# coding: utf-8

# In[385]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
from scipy.stats import pearsonr, spearmanr


# In[386]:


#Reading Cleaned data
train1 = pd.read_csv(r'C:\Users\samhi\OneDrive - sjsu.edu\Documents\sjsu\Fall_24\ML\GWAR\Dataset\cleaned.csv')
print(train1.columns)


# In[387]:


train1.shape[0]


# In[388]:


train2=pd.read_csv(r'C:\Users\samhi\OneDrive - sjsu.edu\Documents\sjsu\Fall_24\ML\GWAR\Dataset\train.csv')


# In[389]:


# List of columns to drop
columns_to_drop = [
    "PAQ_A-Season", "PAQ_A-PAQ_A_Total", "PAQ_C-Season", "PAQ_C-PAQ_C_Total", 
    "PCIAT-Season", "PCIAT-PCIAT_01", "PCIAT-PCIAT_02", "PCIAT-PCIAT_03", 
    "PCIAT-PCIAT_04", "PCIAT-PCIAT_05", "PCIAT-PCIAT_06", "PCIAT-PCIAT_07", 
    "PCIAT-PCIAT_08", "PCIAT-PCIAT_09", "PCIAT-PCIAT_10", "PCIAT-PCIAT_11", 
    "PCIAT-PCIAT_12", "PCIAT-PCIAT_13", "PCIAT-PCIAT_14", "PCIAT-PCIAT_15", 
    "PCIAT-PCIAT_16", "PCIAT-PCIAT_17", "PCIAT-PCIAT_18", "PCIAT-PCIAT_19", 
    "PCIAT-PCIAT_20", "PCIAT-PCIAT_Total", "SDS-Season", "SDS-SDS_Total_Raw", 
    "SDS-SDS_Total_T", "PreInt_EduHx-Season", "PreInt_EduHx-computerinternet_hoursday", "sii"
]

# Drop the columns from your DataFrame
train2= train2.drop(columns=columns_to_drop)

# Check the result
print(train2.columns)


# In[390]:


# List of additional columns to drop
more_columns_to_drop = [
    "CGAS-Season", "Physical-Season", "Physical-Waist_Circumference", 
    "Physical-Diastolic_BP", "Physical-HeartRate", "Physical-Systolic_BP", 
    "Fitness_Endurance-Season", "FGC-Season", "FGC-FGC_CU", "FGC-FGC_GSND", 
    "FGC-FGC_GSD", "FGC-FGC_PU", "FGC-FGC_SRL", "FGC-FGC_SRR", "FGC-FGC_TL", 
    "BIA-Season"
]

# Drop these columns from your DataFrame
train2 = train2.drop(columns=more_columns_to_drop)

# Check the result
print(train2.columns)


# In[391]:


#Checking for number of null values
print(train2.isna().sum())


# In[392]:


#Merging data 
import pandas as pd

# Perform a left join to retain only rows present in train1
train = pd.merge(train1, train2, on='id', how='left')

# Loop through the columns to identify and handle duplicates
columns_to_drop = []

for column in train.columns:
    # Check for columns with the '_x' and '_y' suffix
    if '_x' in column:
        corresponding_y_column = column.replace('_x', '_y')
        if corresponding_y_column in train.columns:
            # If both '_x' and '_y' columns exist, decide which one to keep
            # For this example, we'll keep the '_x' version and drop the '_y' version
            columns_to_drop.append(corresponding_y_column)

# Drop the identified duplicate columns
train = train.drop(columns=columns_to_drop)

# Remove the '_x' suffix from column names to clean up
train.columns = [col.replace('_x', '') if '_x' in col else col for col in train.columns]

# Check the result
print(train.columns)


# In[393]:


#Checking for number of null values
print(train.isna().sum())


# In[394]:


train.shape[0]


# In[395]:


null_CGAS = train['CGAS-CGAS_Score'].isnull().sum()
print('Null values in CGAS-CGAS_Score', null_CGAS)


# In[396]:


import pandas as pd
from sklearn.impute import KNNImputer

# Initialize the KNN imputer with a specified number of neighbors
knn_imputer = KNNImputer(n_neighbors=5)

# Select only the 'CGAS-CGAS_Score' column for imputation
cgas_column = ['CGAS-CGAS_Score']

# Perform KNN imputation only on the 'CGAS-CGAS_Score' column
train[cgas_column] = knn_imputer.fit_transform(train[cgas_column])

# To check if the imputation for 'CGAS-CGAS_Score' was successful
print(train['CGAS-CGAS_Score'].isna().sum())


# In[397]:


#Checking for number of null values
print(train.isna().sum())


# In[398]:


#Handling Fitness Endurance Data

# List of additional columns to drop
Fitnessvitals_columns_to_drop = [
    "Fitness_Endurance-Max_Stage", "Fitness_Endurance-Time_Mins", "Fitness_Endurance-Time_Sec"
]
# Drop these columns from your DataFrame
train = train.drop(columns=Fitnessvitals_columns_to_drop)

# Check the result
print(train.columns)


# #### Fitness Gram Data

# In[399]:


# Convert the 3-category columns into binary, leaving NaN values as is
train['FGC-FGC_GSND_Zone_Binary'] = train['FGC-FGC_GSND_Zone'].apply(
    lambda x: 1 if x == 3 else (0 if x in [1, 2] else None)
)

train['FGC-FGC_GSD_Zone_Binary'] = train['FGC-FGC_GSD_Zone'].apply(
    lambda x: 1 if x == 3 else (0 if x in [1, 2] else None)
)

# Replace the original columns with the binary versions
train['FGC-FGC_GSND_Zone'] = train['FGC-FGC_GSND_Zone_Binary']
train['FGC-FGC_GSD_Zone'] = train['FGC-FGC_GSD_Zone_Binary']

# Drop the temporary binary columns if you no longer need them
train.drop(columns=['FGC-FGC_GSND_Zone_Binary', 'FGC-FGC_GSD_Zone_Binary'], inplace=True)

# Print to verify the changes
print(train[['FGC-FGC_GSND_Zone']])
print(train[['FGC-FGC_GSD_Zone']])


# In[400]:


print(train.columns)


# In[401]:


# List of columns to check for missing values
columns_to_check = [
    'FGC-FGC_CU_Zone',
    'FGC-FGC_GSND_Zone',
    'FGC-FGC_GSD_Zone',
    'FGC-FGC_PU_Zone',
    'FGC-FGC_SRL_Zone',
    'FGC-FGC_SRR_Zone',
    'FGC-FGC_TL_Zone'
]

# Check for missing values in these columns
missing_values = train[columns_to_check].isna().sum()

# Print the count of missing values for each column
print(missing_values)


# In[402]:


# Dropping FGC-FGC_GSND_Zone & FGC-FGC_GSD_Zone columns 

# List of additional columns to drop
Fitnessvitals_columns_to_drop = [
    "FGC-FGC_GSND_Zone","FGC-FGC_GSD_Zone"
]
# Drop these columns from your DataFrame
train = train.drop(columns=Fitnessvitals_columns_to_drop)

# Check the result
print(train.columns)


# In[403]:


import pandas as pd

# Step 1: Calculate the combined score only if all specified columns are non-NaN
train['Combined_Score'] = train.apply(
    lambda row: row[['FGC-FGC_CU_Zone', 'FGC-FGC_PU_Zone', 
                    'FGC-FGC_SRL_Zone', 'FGC-FGC_SRR_Zone', 
                    'FGC-FGC_TL_Zone']].sum() 
    if row[['FGC-FGC_CU_Zone', 'FGC-FGC_PU_Zone', 
            'FGC-FGC_SRL_Zone', 'FGC-FGC_SRR_Zone', 
            'FGC-FGC_TL_Zone']].isna().any() == False 
    else float('nan'), axis=1)

# Step 2: Normalize the 'Combined_Score' to a 0-1 scale
# Remove rows with NaN in 'Combined_Score' before normalization
train_clean = train.dropna(subset=['Combined_Score'])

# Apply Min-Max normalization
min_score = train_clean['Combined_Score'].min()
max_score = train_clean['Combined_Score'].max()

train['Normalized_Combined_Score'] = train['Combined_Score'].apply(
    lambda x: (x - min_score) / (max_score - min_score) if pd.notna(x) else x
)

# Step 3: Drop the original columns and replace them with 'Normalized_Combined_Score'
train.drop(columns=['FGC-FGC_CU_Zone', 'FGC-FGC_PU_Zone', 
                    'FGC-FGC_SRL_Zone', 'FGC-FGC_SRR_Zone', 
                    'FGC-FGC_TL_Zone'], inplace=True)

# Replace the original columns with 'Normalized_Combined_Score' in the data
train['Normalized_Combined_Score'] = train['Normalized_Combined_Score']

# Print summary to verify
print(train.head())


# In[404]:


null_FCombined = train['Normalized_Combined_Score'].isnull().sum()
print('Null values in null_FCombined', null_FCombined)


# In[405]:


#knn on fitnessgram combined score. 

from sklearn.impute import KNNImputer

# Initialize the KNN imputer with a specified number of neighbors
knn_imputer = KNNImputer(n_neighbors=5)

# Apply KNN imputation to the 'Normalized_Combined_Score' column
train['Normalized_Combined_Score'] = knn_imputer.fit_transform(train[['Normalized_Combined_Score']])

# Check if imputation was successful by printing the count of missing values again
missing_values_after_imputation = train['Normalized_Combined_Score'].isna().sum()
print(f"Missing values after imputation: {missing_values_after_imputation}")


# In[406]:


#Understanding the correlation between sii and Normalised combined score
# Calculate the correlation between 'Normalized_Combined_Score' and 'SII'
correlation = train['Normalized_Combined_Score'].corr(train['sii'])

# Print the correlation value
print(f"Correlation between Normalized_Combined_Score and SII: {correlation}")


# In[407]:


# Rename the 'Normalized_Combined_Score' column to 'Fitness_Combined_Score'
train.rename(columns={'Normalized_Combined_Score': 'Fitness_Combined_Score'}, inplace=True)

# Print the updated DataFrame to verify
print(train.head())


# ### Bio-Electric Impedance Data

# In[408]:


# Drop these columns from your DataFrame
train = train.drop(columns="BIA-BIA_FFM")

# Check the result
print(train.columns)


# In[409]:


import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Specify the columns to normalize and their weights
columns_to_normalize = [
    'BIA-BIA_BMR', 'BIA-BIA_DEE', 'BIA-BIA_SMM', 'BIA-BIA_Fat',
    'BIA-BIA_BMI', 'BIA-BIA_TBW', 'BIA-BIA_ICW', 'BIA-BIA_ECW',
    'BIA-BIA_LST', 'BIA-BIA_LDM'
]

weights = {
    'BIA-BIA_BMR': 0.25,  # High importance
    'BIA-BIA_DEE': 0.25,  # High importance
    'BIA-BIA_SMM': 0.15,  # Moderate importance
    'BIA-BIA_Fat': 0.15,  # Moderate importance
    'BIA-BIA_BMI': 0.1,   # Moderate importance
    'BIA-BIA_TBW': 0.05,  # Low importance
    'BIA-BIA_ICW': 0.05,  # Low importance
    'BIA-BIA_ECW': 0.05,  # Low importance
    'BIA-BIA_LST': 0.025, # Least importance
    'BIA-BIA_LDM': 0.025  # Least importance
}

# Extract only the columns to be normalized
train_subset = train[columns_to_normalize]

# Normalizing the selected columns using Min-Max scaling
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(train_subset)
train_normalized = pd.DataFrame(normalized_data, columns=columns_to_normalize)

# Calculate the composite index using the specified weights
train_normalized['Composite_Index'] = sum(
    train_normalized[column] * weight for column, weight in weights.items()
)

# Add the 'Composite_Index' to the original 'train' DataFrame as 'Physical_Composite_Index'
train['Physical_Composite_Index'] = train_normalized['Composite_Index']

# Output the resulting DataFrame with the new column
print(train.columns)


# In[410]:


missing_values = train['Physical_Composite_Index'].isna().sum()
print(missing_values)


# In[411]:


# Initialize the KNN imputer, specify the number of neighbors (k)
imputer = KNNImputer(n_neighbors=5)  # You can adjust n_neighbors as needed

# Apply KNN imputation on the 'Physical_Composite_Index' column
train['Physical_Composite_Index'] = imputer.fit_transform(train[['Physical_Composite_Index']])

# Output the DataFrame to check the filled values
print(train[['Physical_Composite_Index']].head())


# In[412]:


import pandas as pd
from scipy.stats import pearsonr, spearmanr

# Assuming 'train' is your existing DataFrame
# Calculate Pearson correlation (linear, parametric)
pearson_corr, _ = pearsonr(train['Physical_Composite_Index'], train['sii'])
print(f"Pearson Correlation between Physical_Composite_Index and SII: {pearson_corr}")

# Calculate Spearman correlation (non-parametric)
spearman_corr, _ = spearmanr(train['Physical_Composite_Index'], train['sii'])
print(f"Spearman Correlation between Physical_Composite_Index and SII: {spearman_corr}")


# In[413]:


from scipy.stats import kendalltau
from sklearn.feature_selection import mutual_info_regression
import numpy as np
from dcor import distance_correlation

# Assuming 'train' is your existing DataFrame

# 1. Kendall's Tau (Non-Linear Measure)
kendall_corr, _ = kendalltau(train['Physical_Composite_Index'], train['sii'])
print(f"Kendall's Tau Correlation between Physical_Composite_Index and SII: {kendall_corr}")

# 2. Mutual Information (Non-Linear Measure)
# We need to reshape the data to use mutual_info_regression
X = train[['Physical_Composite_Index']]
y = train['sii']

mutual_info = mutual_info_regression(X, y)
print(f"Mutual Information between Physical_Composite_Index and SII: {mutual_info[0]}")

# 3. Distance Correlation (Non-Linear Measure)
# Distance Correlation can capture non-linear relationships
dcor_value = distance_correlation(train['Physical_Composite_Index'], train['sii'])
print(f"Distance Correlation between Physical_Composite_Index and SII: {dcor_value}")


# In[414]:


# Dropping BIA columns

columns_to_drop = [
    'BIA-BIA_BMR', 'BIA-BIA_DEE', 'BIA-BIA_SMM', 'BIA-BIA_Fat','Physical-Height', 'Physical-Weight',
    'BIA-BIA_BMI', 'BIA-BIA_TBW', 'BIA-BIA_ICW', 'BIA-BIA_ECW','BIA-BIA_LST', 'BIA-BIA_Frame_num',                         
    'BIA-BIA_LDM', 'Combined_Score', 'BIA-BIA_BMC','BIA-BIA_FFMI','BIA-BIA_FMI'
]

# Drop these columns from your DataFrame
train = train.drop(columns=columns_to_drop)

# Check the result
print(train.columns)


# In[415]:


# Handling Basic_Demos-Enroll_Season's empty values
# Impute missing values in 'Basic_Demos-Enroll_Season' with mode
mode_value = train['Basic_Demos-Enroll_Season'].mode()[0] 
train['Basic_Demos-Enroll_Season'].fillna(mode_value, inplace=True)

# Verifying if imputation is successful.
print(train['Basic_Demos-Enroll_Season'].isnull().sum())  


# In[416]:


#Checking for number of null values
print(train.isna().sum())


# In[417]:


# Initialize KNNImputer with the number of neighbors to consider
knn_imputer = KNNImputer(n_neighbors=5)

# Apply KNN imputation and update 'Physical-BMI' in the original DataFrame
train[['Physical-BMI']] = knn_imputer.fit_transform(train[['Physical-BMI']])

# Check if missing values are imputed in 'Physical-BMI'
print(train['Physical-BMI'].isnull().sum())  


# In[418]:


#Checking for number of null values
print(train.isna().sum())


# In[420]:


# Check distribution of non-null values
print(train['BIA-BIA_Activity_Level_num'].value_counts(normalize=True))

# Check if missing values are random or related to other variables
import seaborn as sns
import matplotlib.pyplot as plt

# Analyze missing vs non-missing patterns with other variables
train['activity_missing'] = train['BIA-BIA_Activity_Level_num'].isna().astype(int)


# In[423]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class ActivityLevelImputer:
    def __init__(self, df):
        self.df = df.copy()
        self.original_distribution = {
            1.0: 0.129950,
            2.0: 0.317903,
            3.0: 0.349136,
            4.0: 0.158951,
            5.0: 0.044060
        }
    
    def calculate_physical_score(self):
        """Calculate a physical score based on available metrics"""
        physical_features = [
            'Physical-BMI',
            'Fitness_Combined_Score',
            'Physical_Composite_Index'
        ]
        
        # Normalize features
        scaler = MinMaxScaler()
        normalized = pd.DataFrame(
            scaler.fit_transform(self.df[physical_features]),
            columns=physical_features
        )
        
        # Calculate composite score
        return normalized.mean(axis=1)
    
    def impute_activity_levels(self):
        """Impute missing activity levels maintaining original distribution"""
        # Calculate physical score for all rows
        physical_scores = self.calculate_physical_score()
        
        # Impute missing values directly
        imputed_values = self.df['BIA-BIA_Activity_Level_num'].copy()
        
        missing_count = imputed_values.isna().sum()
        
        if missing_count > 0:
            # Calculate number of values needed for each level
            level_counts = {
                level: int(np.round(prop * missing_count))
                for level, prop in self.original_distribution.items()
            }
            
            # Adjust for rounding errors
            total_allocated = sum(level_counts.values())
            if total_allocated < missing_count:
                level_counts[3.0] += missing_count - total_allocated
            
            # Sort missing values by physical score
            missing_scores = physical_scores[imputed_values.isna()]
            sorted_indices = missing_scores.sort_values().index
            
            # Allocate values maintaining distribution
            new_values = []
            for level in sorted([1.0, 2.0, 3.0, 4.0, 5.0]):
                count = level_counts[level]
                new_values.extend([level] * count)
            
            # Assign imputed values to the original DataFrame
            imputed_values[sorted_indices] = new_values
        
        return imputed_values
    
    def validate_imputation(self, imputed_series):
        """Validate the imputation results"""
        # Compare distributions
        original_dist = self.df['BIA-BIA_Activity_Level_num'].value_counts(normalize=True)
        imputed_dist = imputed_series.value_counts(normalize=True)
        
        # Calculate correlations with physical metrics
        correlations = {
            'BMI': imputed_series.corr(self.df['Physical-BMI']),
            'Fitness': imputed_series.corr(self.df['Fitness_Combined_Score']),
            'Physical_Composite': imputed_series.corr(self.df['Physical_Composite_Index'])
        }
        
        return {
            'original_distribution': original_dist,
            'imputed_distribution': imputed_dist,
            'correlations': correlations
        }


# Initialize imputer
imputer = ActivityLevelImputer(train)

# Perform imputation
imputed_activity = imputer.impute_activity_levels()

# Validate results
validation = imputer.validate_imputation(imputed_activity)

print("Original Distribution:")
print(validation['original_distribution'])
print("\nImputed Distribution:")
print(validation['imputed_distribution'])
print("\nCorrelations with Physical Metrics:")
print(validation['correlations'])

# Update the dataframe
train['BIA-BIA_Activity_Level_num'] = imputed_activity


# In[424]:


print(train.isna().sum())


# In[425]:


print(train.shape)


# In[426]:


# Rename the columns
train.rename(columns={
    'BIA-BIA_Activity_Level_num': 'BIA_Activity_Level',
    'CGAS-CGAS_Score': 'CGAS_Score'
}, inplace=True)

# Remove the 'activity_missing' column
train.drop(columns=['activity_missing'], inplace=True)

# Reorder the columns as specified
column_order = [
    'id',
    'Basic_Demos-Age',
    'Basic_Demos-Sex',
    'Basic_Demos-Enroll_Season',
    'age_group',
    'CGAS_Score',
    'Physical-BMI',
    'BIA_Activity_Level',
    'Fitness_Combined_Score',
    'Physical_Composite_Index',
    'SDS-SDS_Total_T',
    'PreInt_EduHx-computerinternet_hoursday',
    'sii'
]

train = train[column_order]


# In[430]:


# Select numerical columns
numerical_columns = train.select_dtypes(include=['number']).columns

# Calculate the range (min and max) for each numerical column
range_values = train[numerical_columns].agg(['min', 'max'])

# Print the range for each numerical column
print(range_values)


# In[431]:


# Drop these columns from your DataFrame
train = train.drop(columns='Basic_Demos-Enroll_Season')


# In[432]:


# Save the DataFrame to a CSV
train.to_csv('cleaned_train.csv', index=False)

