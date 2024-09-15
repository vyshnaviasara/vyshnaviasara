#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd

# Reading a CSV file
data = pd.read_csv('/Users/vyshnaviasara/Downloads/202101-tripdata.csv')

# Display the first few rows of the dataframe
print(data.head())


# In[3]:


data.dropna(inplace=True)


# In[ ]:


from sklearn.model_selection import train_test_split

X = data.drop('target_column', axis=1)  # Features
y = data['target_column']  # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[ ]:


# Import module
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[6]:


data.info(verbose=True)


# In[8]:


non_null_counts = data.notnull().sum()
print(non_null_counts)


# In[ ]:


#DATA CLEANING


# In[9]:


# Columns to be dropped
columns_to_drop = ["ride_id", 
                   "start_station_name", "start_station_id", 
                   "end_station_name", "end_station_id", 
                   "start_lat", "start_lng", 
                   "end_lat", "end_lng"]

# Drop the columns from the DataFrame
data = data.drop(columns=columns_to_drop, axis=1)

# Verify the columns have been dropped
data.info()


# In[10]:


non_null_counts_update = data.notnull().sum()
non_null_counts_update


# In[ ]:


#outlier


# In[11]:


# Check if invalid value exist in columns "member_casual"
invalid_user_types_count = data[~data["member_casual"].isin(["casual", "member"])].shape[0]
invalid_user_types_count


# In[13]:


# Check if invalid value exist in columns "rideable_type"
valid_rideable_types = ["electric_bike", "classic_bike", "docked_bike"]
invalid_rideable_types_count = data[~data["rideable_type"].isin(valid_rideable_types)].shape[0]
invalid_rideable_types_count


# In[14]:


# Transfer from string format to time format
data["started_at"] = pd.to_datetime(data["started_at"])
data["ended_at"] = pd.to_datetime(data["ended_at"])

# Count the number of trips where the end time is not greater than the start time
invalid_durations = data[data["ended_at"] <= data["started_at"]].index
invalid_durations


# In[15]:


# Delete outliers values
data.drop(index= invalid_durations, inplace= True)


# In[17]:


#Create a new columns call "Trip Duration"
data["Trip Duration"] = data["ended_at"] - data["started_at"]
data["Trip Duration"]


# In[18]:


print(data["Trip Duration"].max())
print(data["Trip Duration"].min())


# In[19]:


# Define the minimum and maximum durations in minutes
min_duration = pd.Timedelta("5 minutes")
max_duration = pd.Timedelta("12 hours")

# Filter the DataFrame for trips with durations within the specified range
data_filtered = data[(data["Trip Duration"] >= min_duration) & (data["Trip Duration"] <= max_duration)]

# df_filtered contains only the trips that are between 5 minutes and 12 hours


# In[ ]:


3.3.3 Duplicates



# In[21]:


duplicates = data_filtered.duplicated()

num_duplicates = duplicates.sum()
num_duplicates


# In[22]:


# Delete duplicated
data_filtered.drop(index= num_duplicates, inplace = True)


# In[23]:


non_null_counts_v2 = data.notnull().sum()
non_null_counts_v2


# In[ ]:


Step 4: Data analysis


# In[24]:


number_counts = data_filtered["member_casual"].value_counts()
ratio_counts = data_filtered["member_casual"].value_counts(normalize= True)

print(number_counts)
print()
print(ratio_counts)


# In[25]:


num_rideable = data_filtered["rideable_type"].value_counts()
ratio_rideable = data_filtered["rideable_type"].value_counts(normalize= True)

print(num_rideable)
print()
print(ratio_rideable)


# In[26]:


rideable_user_ratio = data_filtered.groupby("rideable_type")["member_casual"].value_counts(normalize=True).unstack()
rideable_user_ratio


# In[ ]:


Trip duration


# In[27]:


# Calculate Trip Duration for BOTH casual and member user
data["Trip Duration(Casual)"] = data_filtered["Trip Duration"].where(data["member_casual"] == "casual")
data["Trip Duration(Member)"] = data_filtered["Trip Duration"].where(data["member_casual"] == "member")

print(data["Trip Duration(Casual)"])
print(data["Trip Duration(Member)"])


# In[29]:


# Calculate Average Trip Duration for BOTH casual and member user
casual_avg_duration = data_filtered["Trip Duration"].where(data["member_casual"] == "casual", pd.NaT).mean()
member_avg_duration = data_filtered["Trip Duration"].where(data["member_casual"] == "member", pd.NaT).mean()
# Print result
print(casual_avg_duration)
print(member_avg_duration)


# In[31]:


# Calculate average ride time for casual users by month
casual_avg_by_month = data_filtered[data_filtered["member_casual"] == "casual"].groupby(pd.Grouper(key="started_at", freq="M"))["Trip Duration"].mean()

# Calculate average ride time for member users by month
member_avg_by_month = data_filtered[data_filtered["member_casual"] == "member"].groupby(pd.Grouper(key="started_at", freq="M"))["Trip Duration"].mean()

print(casual_avg_by_month)
print(member_avg_by_month)


# In[ ]:


4.3 Day of Week


# In[33]:


# Create a new column "day_of_week"
data_filtered["Day_of_week"] = data_filtered["started_at"].dt.dayofweek
# Filter out weekend((Saturday=5, Sunday=6))
weekday_data = data_filtered[data_filtered["Day_of_week"] < 5]
weekday_data["Day_of_week"]


# In[34]:


# Calculate Day of week for BOTH casual and member user
weekday_data["Day of Week(Casual)"] = weekday_data["Day_of_week"].where(data_filtered["member_casual"] == "casual", pd.NaT)
weekday_data["Day of Week(Member)"] = weekday_data["Day_of_week"].where(data_filtered["member_casual"] == "member", pd.NaT)


# In[ ]:


# Save file
# df.to_csv("Cyclistic_Trip_Data.csv")


# In[35]:


data_filtered.describe()


# In[ ]:


Step 5 Data Visualization and Conclusion


# In[ ]:


5.1 Proportion


# In[37]:


import matplotlib.pyplot as plt


# In[38]:


plt.pie(ratio_counts, labels=ratio_counts.index, autopct="%1.1f%%", startangle=90)
plt.title("Proportion of Rides by User Type")
plt.show()


# In[39]:


plt.pie(ratio_rideable, labels=ratio_rideable.index, autopct="%1.1f%%", startangle=90)
plt.title("Proportion of Rides by Rideable Type")
plt.show()


# In[41]:


import numpy as np


# In[42]:


rideable_user_ratio.plot.bar(stacked=True)

ind = np.arange(len(rideable_user_ratio))
plt.xticks(ind, rideable_user_ratio.index, rotation="horizontal")

plt.title("Proportion of Casual and Member Users for Each Rideable Type")
plt.xlabel("Rideable Type")
plt.ylabel("Proportion of Users")


# In[43]:


# Convert time into minutes
casual_avg_minutes = casual_avg_duration.total_seconds() / 60
member_avg_minutes = member_avg_duration.total_seconds() / 60

labels = ["Casual Users", "Member Users"]
average_durations = [casual_avg_minutes, member_avg_minutes]

plt.bar(labels, average_durations)

plt.xlabel("User Type")
plt.ylabel("Average Duration (minutes)")
plt.title("Average Trip Duration for both types of user")

plt.show()


# In[44]:


# Convert the average trip durations from timedelta to minutes
casual_avg_minutes_by_month = casual_avg_by_month.dt.total_seconds() / 60
member_avg_minutes_by_month = member_avg_by_month.dt.total_seconds() / 60

months = casual_avg_minutes_by_month.index.strftime("%Y-%m")  # Formatting the month as Year-Month
casual_averages = casual_avg_minutes_by_month.values
member_averages = member_avg_minutes_by_month.values

bar_width = 0.35

# Set position of bar on X axis
r1 = np.arange(len(casual_averages))
r2 = [x + bar_width for x in r1]

# Make the plot
plt.figure(figsize=(10, 6))
plt.bar(r1, casual_averages, color="blue", width=bar_width, edgecolor="grey", label="Casual Users")
plt.bar(r2, member_averages, color="green", width=bar_width, edgecolor="grey", label="Member Users")

# Add xticks on the middle of the group bars
plt.xlabel("Month", fontweight="bold")
plt.xticks([r + bar_width for r in range(len(casual_averages))], months)

plt.title("Average Ride Time by User Type and Month")
plt.ylabel("Average Duration (minutes)")
plt.legend()

plt.show()


# In[49]:


import pandas as pd
import matplotlib.pyplot as plt

# Sample data
data = pd.DataFrame({
    'date': ['2024-07-21', '2024-07-22', '2024-07-23', '2024-07-24', '2024-07-25', '2024-07-26', '2024-07-27'],
    'ride_id': [1, 2, 3, 4, 5, 6, 7]
})

# Convert the 'date' column to datetime
data['date'] = pd.to_datetime(data['date'])

# Extract the day of the week: 0=Monday, 1=Tuesday, ..., 6=Sunday
data['day_of_week'] = data['date'].dt.dayofweek

# Define weekday and weekend DataFrames
weekday_df = data[data['day_of_week'] < 5]
weekend_df = data[data['day_of_week'] >= 5]

# Calculate the number of rides
num_rides_weekday = weekday_df.shape[0]
num_rides_weekend = weekend_df.shape[0]

# Create labels and corresponding values
labels = ["Weekday", "Weekend"]
day_of_the_week_counts = [num_rides_weekday, num_rides_weekend]

# Create the bar graph
plt.bar(labels, day_of_the_week_counts, color=['blue', 'orange'])

# Set the labels and title
plt.xlabel("Day Type")
plt.ylabel("Number of Rides")
plt.title("Ride Counts: Weekday vs Weekend")

# Display the plot
plt.show()


# In[ ]:


5.3.2 Ride in Weekday for both types of users


# In[51]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Sample data
data = pd.DataFrame({
    'date': ['2024-07-21', '2024-07-22', '2024-07-23', '2024-07-24', '2024-07-25', '2024-07-26', '2024-07-27',
             '2024-07-28', '2024-07-29', '2024-07-30'],
    'user_type': ['Casual', 'Member', 'Casual', 'Member', 'Casual', 'Member', 'Casual', 'Member', 'Casual', 'Member']
})

# Convert the 'date' column to datetime and extract the day of the week
data['date'] = pd.to_datetime(data['date'])
data['day_of_week'] = data['date'].dt.dayofweek  # 0=Monday, 1=Tuesday, ..., 6=Sunday

# Separate weekday data
weekday_df = data[data['day_of_week'] < 5]  # Monday (0) to Friday (4)

# Count the number of rides for casual and member users on each weekday
weekday_casual_counts = weekday_df[weekday_df['user_type'] == 'Casual']['day_of_week'].value_counts().sort_index()
weekday_member_counts = weekday_df[weekday_df['user_type'] == 'Member']['day_of_week'].value_counts().sort_index()

# Prepare the data for plotting
days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
casual_counts = [weekday_casual_counts.get(i, 0) for i in range(5)]
member_counts = [weekday_member_counts.get(i, 0) for i in range(5)]

# Define the width of the bars
bar_width = 0.35

# Set position of bar on X axis
r1 = np.arange(len(casual_counts))
r2 = [x + bar_width for x in r1]

# Create the bar graph
plt.figure(figsize=(11, 6))
plt.bar(r1, casual_counts, color="blue", width=bar_width, label="Casual Users")
plt.bar(r2, member_counts, color="orange", width=bar_width, label="Member Users")

# Add xticks on the middle of the group bars
plt.xlabel("Day of the Week", fontweight="bold")
plt.xticks([r + bar_width / 2 for r in range(len(casual_counts))], days)

plt.title("Number of Rides by User Type on Weekdays")
plt.ylabel("Number of Rides")
plt.legend()

plt.show()


# In[ ]:


5.3.3 Ride in Weekend for both types of users


# In[54]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Sample data - replace with your actual data
data = pd.DataFrame({
    'date': ['2024-07-21', '2024-07-22', '2024-07-23', '2024-07-24', '2024-07-25', '2024-07-26', '2024-07-27',
             '2024-07-28', '2024-07-29', '2024-07-30'],
    'user_type': ['Casual', 'Member', 'Casual', 'Member', 'Casual', 'Member', 'Casual', 'Member', 'Casual', 'Member']
})

# Convert the 'date' column to datetime and extract the day of the week
data['date'] = pd.to_datetime(data['date'])
data['day_of_week'] = data['date'].dt.dayofweek  # 0=Monday, 1=Tuesday, ..., 6=Sunday

# Check for the correct columns
if 'user_type' not in data.columns or 'day_of_week' not in data.columns:
    raise ValueError("The dataframe must contain 'user_type' and 'day_of_week' columns.")

# Separate weekday data
weekday_df = data[data['day_of_week'] < 5]  # Monday (0) to Friday (4)

# Count the number of rides for casual and member users on each weekday
weekday_casual_counts = weekday_df[weekday_df['user_type'] == 'Casual']['day_of_week'].value_counts().sort_index()
weekday_member_counts = weekday_df[weekday_df['user_type'] == 'Member']['day_of_week'].value_counts().sort_index()

# Prepare the data for plotting
days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
casual_counts = [weekday_casual_counts.get(i, 0) for i in range(5)]
member_counts = [weekday_member_counts.get(i, 0) for i in range(5)]

# Define the width of the bars
bar_width = 0.35

# Set position of bar on X axis
r1 = np.arange(len(casual_counts))
r2 = [x + bar_width for x in r1]

# Create the bar graph
plt.figure(figsize=(11, 6))
plt.bar(r1, casual_counts, color="blue", width=bar_width, label="Casual Users")
plt.bar(r2, member_counts, color="orange", width=bar_width, label="Member Users")

# Add xti


# In[ ]:




