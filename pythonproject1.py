#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv('COVID.csv')


# In[3]:


print(df.columns)
df.head()


# In[4]:


df['Week End'] = pd.to_datetime(df['Week End'])

# Sort by date
df = df.sort_values('Week End')

# (Optional) Filter for only 'All' age group to reduce noise
df_all = df[df['Age Group'] == 'All']

# Plotting
plt.figure(figsize=(14, 6))
plt.plot(df_all['Week End'], df_all['Unvaccinated Rate'], label='Unvaccinated Rate', color='red')
plt.plot(df_all['Week End'], df_all['Vaccinated Rate'], label='Vaccinated Rate', color='green')
plt.plot(df_all['Week End'], df_all['Boosted Rate'], label='Boosted Rate', color='blue')

plt.title('COVID-19 Case Rates by Vaccination Status Over Time')
plt.xlabel('Week End Date')
plt.ylabel('Case Rate')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# In[5]:


df['Week End'] = pd.to_datetime(df['Week End'])

# Filter to only "All" age group and "Deaths" outcome
df_deaths = df[(df['Age Group'] == 'All') & (df['Outcome'] == 'Deaths')]

# Drop rows with missing values in the rates
df_deaths = df_deaths.dropna(subset=['Unvaccinated Rate', 'Vaccinated Rate', 'Boosted Rate'])

# Calculate average rates over time
avg_unvacc = df_deaths['Unvaccinated Rate'].mean()
avg_vacc = df_deaths['Vaccinated Rate'].mean()
avg_boosted = df_deaths['Boosted Rate'].mean()

# Create a pie chart
labels = ['Unvaccinated', 'Vaccinated', 'Boosted']
sizes = [avg_unvacc, avg_vacc, avg_boosted]
colors = ['red', 'green', 'blue']

plt.figure(figsize=(7, 7))
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
plt.title('Average Death Rate by Vaccination Status')
plt.axis('equal')
plt.show()


# In[7]:


df['Week End'] = pd.to_datetime(df['Week End'])

# Use only rows with Outcome = 'Cases' and Age Group = 'All'
df_cases = df[(df['Outcome'] == 'Cases') & (df['Age Group'] == 'All')]
df_cases = df_cases.dropna(subset=['Unvaccinated Rate'])

# Convert dates to ordinal (for numerical regression)
df_cases['Date_Ordinal'] = df_cases['Week End'].map(pd.Timestamp.toordinal)

# X (dates as numbers) and y (unvaccinated case rate)
X = df_cases['Date_Ordinal'].values
y = df_cases['Unvaccinated Rate'].values

# Manual linear regression: compute slope (m) and intercept (b)
x_mean = np.mean(X)
y_mean = np.mean(y)
m = np.sum((X - x_mean) * (y - y_mean)) / np.sum((X - x_mean)**2)
b = y_mean - m * x_mean

# Predict existing values
y_pred = m * X + b

# Predict future (next 10 weeks)
future_dates = pd.date_range(start=df_cases['Week End'].max(), periods=10, freq='W')
future_ordinals = future_dates.map(pd.Timestamp.toordinal).values
future_preds = m * future_ordinals + b

# Plotting
plt.figure(figsize=(14, 6))
plt.plot(df_cases['Week End'], y, label='Actual Cases (Unvaccinated)', color='blue')
plt.plot(df_cases['Week End'], y_pred, label='Predicted (Trend)', color='orange', linestyle='--')
plt.plot(future_dates, future_preds, label='Future Surge Prediction', color='red', linestyle='dotted')

plt.title('Manual Prediction of COVID-19 Case Surges (Unvaccinated)')
plt.xlabel('Week End Date')
plt.ylabel('Case Rate')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# Blue line = actual past data
# 
# Orange dashed = predicted trend line (manual regression)
# 
# Red dotted = forecasted surge for next 10 weeks

# In[9]:


df['Week End'] = pd.to_datetime(df['Week End'])

# Filter for Age Group 'All' and Outcome 'Deaths'
df_deaths = df[(df['Age Group'] == 'All') & (df['Outcome'] == 'Deaths')]

# Drop missing values
df_deaths = df_deaths.dropna(subset=['Unvaccinated Rate', 'Vaccinated Rate', 'Boosted Rate'])

# Set plot style
sns.set(style='whitegrid')

# Plot histogram
plt.figure(figsize=(12, 6))
sns.histplot(df_deaths['Unvaccinated Rate'], color='red', label='Unvaccinated', kde=True, stat='density', bins=15)
sns.histplot(df_deaths['Vaccinated Rate'], color='green', label='Vaccinated', kde=True, stat='density', bins=15)
sns.histplot(df_deaths['Boosted Rate'], color='blue', label='Boosted', kde=True, stat='density', bins=15)

plt.title('Distribution of COVID-19 Mortality Rates by Vaccination Status')
plt.xlabel('Mortality Rate')
plt.ylabel('Density')
plt.legend()
plt.tight_layout()
plt.show()


# In[ ]:





# In[14]:


# 2. Pie Chart – Outcome Distribution
plt.figure(figsize=(8,8))
outcome_counts = df['Outcome'].value_counts()
plt.pie(outcome_counts, labels=outcome_counts.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette("Set2"))
plt.title('COVID Outcomes Distribution')
plt.show()



# In[15]:


# 3. Line Chart – Unvaccinated Rate Trends by Age Group
plt.figure(figsize=(14,7))
age_groups = df['Age Group'].unique()
for group in age_groups:
    subset = df[df['Age Group'] == group]
    if not subset['Unvaccinated Rate'].isna().all():
        plt.plot(subset['Week End'], subset['Unvaccinated Rate'], label=group)

plt.title('Unvaccinated Rate Over Time by Age Group')
plt.xlabel('Week End')
plt.ylabel('Unvaccinated Rate')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[16]:


# 4. Boxplot – Spread of Vaccination Rates
plt.figure(figsize=(12,6))
sns.boxplot(data=df[['Vaccinated Rate', 'Boosted Rate', 'Unvaccinated Rate']])
plt.title('Distribution of Vaccination Rates')
plt.ylabel('Rate')
plt.tight_layout()
plt.show()


# In[17]:


#Analyze the relationship between vaccination status and COVID-19 outcomes using correlation analysis.
corr_columns = [
    'Unvaccinated Rate', 'Vaccinated Rate', 'Boosted Rate',
    'Crude Vaccinated Ratio', 'Crude Boosted Ratio',
    'Age-Adjusted Vaccinated Rate', 'Age-Adjusted Boosted Rate',
    'Age-Adjusted Unvaccinated Rate'
]

# Clean data: drop rows with NaN in selected columns
corr_df = df[corr_columns].dropna()

# Compute correlation matrix
corr_matrix = corr_df.corr()

# Plot heatmap
plt.figure(figsize=(12,8))
sns.heatmap(corr_matrix, annot=True, cmap='YlGnBu', linewidths=0.5, fmt='.2f')
plt.title('Correlation Between Vaccination Metrics and Outcomes')
plt.tight_layout()
plt.show()


# In[18]:


#Compare the number of COVID-19 deaths across different age groups to identify the most affected demographic segments.
# Filter for 'Deaths' outcome only
deaths_df = df[df['Outcome'] == 'Deaths']

# Group by Age Group and count deaths
death_counts = deaths_df['Age Group'].value_counts().sort_index()

# Plotting bar chart
plt.figure(figsize=(12,6))
sns.barplot(x=death_counts.index, y=death_counts.values, palette='coolwarm')
plt.title('COVID-19 Deaths by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Number of Deaths')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[19]:


#Compare the impact of booster doses on death outcomes across different age groups.
deaths = df[df['Outcome'] == 'Deaths']

# Group by Age Group
boosted_vs_death = deaths.groupby('Age Group').agg({
    'Boosted Rate': 'mean',
    'Outcome': 'count'  # Counting death cases
}).rename(columns={'Outcome': 'Total Deaths'}).reset_index()

# Plotting
fig, ax1 = plt.subplots(figsize=(12,6))

# Bar chart for deaths
sns.barplot(data=boosted_vs_death, x='Age Group', y='Total Deaths', color='salmon', label='Total Deaths')

# Line plot (secondary y-axis) for boosted rate
ax2 = ax1.twinx()
sns.lineplot(data=boosted_vs_death, x='Age Group', y='Boosted Rate', marker='o', sort=False, ax=ax2, color='green', label='Avg Boosted Rate')

# Titles and labels
ax1.set_title('Booster Rate vs COVID Deaths by Age Group')
ax1.set_xlabel('Age Group')
ax1.set_ylabel('Total Deaths')
ax2.set_ylabel('Average Boosted Rate')

# Legends
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[20]:


death_data = pd.DataFrame({
    'Vaccinated': df['Age-Adjusted Vaccinated Rate'],
    'Unvaccinated': df['Age-Adjusted Unvaccinated Rate'],
    'Boosted': df['Age-Adjusted Boosted Rate']
})

# Drop rows with missing values
death_data = death_data.dropna()

# Melt the dataframe for boxplot compatibility
death_melted = death_data.melt(var_name='Status', value_name='Death Rate')

# Plot
plt.figure(figsize=(10,6))
sns.boxplot(x='Status', y='Death Rate', data=death_melted, palette='pastel')
plt.title('Distribution of Age-Adjusted COVID Death Rates by Vaccination Status')
plt.xlabel('Vaccination Status')
plt.ylabel('Death Rate')
plt.tight_layout()
plt.show()


# Examine the distribution and variability of COVID-19 death rates among vaccinated and unvaccinated populations.

# In[ ]:




