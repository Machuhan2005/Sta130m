#!/usr/bin/env python
# coding: utf-8

# In[11]:


#Q1:
import plotly.express as px
import pandas as pd
import numpy as np
import plotly.graph_objects as go

url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv"
penguins = pd.read_csv(url)

# Remove rows with missing values in key columns
penguins = penguins.dropna(subset=['species', 'flipper_length_mm'])

# Function to add lines and rectangles to each subplot
def add_lines_rectangles(fig, species_name, data, col):
    mean_flipper = data['flipper_length_mm'].mean()
    median_flipper = data['flipper_length_mm'].median()
    std_flipper = data['flipper_length_mm'].std()
    min_flipper = data['flipper_length_mm'].min()
    max_flipper = data['flipper_length_mm'].max()
    q1 = data['flipper_length_mm'].quantile(0.25)
    q3 = data['flipper_length_mm'].quantile(0.75)

    # Add lines for mean and median
    fig.add_vline(x=mean_flipper, line=dict(color='blue', dash='dash', width=2),
                  annotation_text="Mean", annotation_position="top left", col=col, row=1)
    fig.add_vline(x=median_flipper, line=dict(color='green', dash='dot', width=2),
                  annotation_text="Median", annotation_position="top right", col=col, row=1)
    
    # Add rectangles for ranges
    fig.add_vrect(x0=min_flipper, x1=max_flipper, fillcolor='lightgrey', opacity=0.2, 
                  line_width=0, annotation_text="Range", col=col, row=1)
    fig.add_vrect(x0=q1, x1=q3, fillcolor='yellow', opacity=0.3, line_width=0, 
                  annotation_text="IQR", col=col, row=1)
    fig.add_vrect(x0=mean_flipper - 2 * std_flipper, x1=mean_flipper + 2 * std_flipper, 
                  fillcolor='red', opacity=0.2, line_width=0, annotation_text="±2σ", col=col, row=1)

# Create a facet plot for histograms of each species
fig = px.histogram(penguins, x='flipper_length_mm', color='species', facet_col='species', 
                   title='Penguin Flipper Lengths by Species', marginal="box", 
                   template="plotly_white")

# Loop through each species and add lines and rectangles
for i, species in enumerate(penguins['species'].unique()):
    species_data = penguins[penguins['species'] == species]
    add_lines_rectangles(fig, species, species_data, col=i + 1)

# Show the plot
fig.show()


# In[11]:


#Q1:
import plotly.express as px
import pandas as pd
import numpy as np
import plotly.graph_objects as go

url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv"
penguins = pd.read_csv(url)

# Remove rows with missing values in key columns
penguins = penguins.dropna(subset=['species', 'flipper_length_mm'])

# Function to add lines and rectangles to each subplot
def add_lines_rectangles(fig, species_name, data, col):
    mean_flipper = data['flipper_length_mm'].mean()
    median_flipper = data['flipper_length_mm'].median()
    std_flipper = data['flipper_length_mm'].std()
    min_flipper = data['flipper_length_mm'].min()
    max_flipper = data['flipper_length_mm'].max()
    q1 = data['flipper_length_mm'].quantile(0.25)
    q3 = data['flipper_length_mm'].quantile(0.75)

    # Add lines for mean and median
    fig.add_vline(x=mean_flipper, line=dict(color='blue', dash='dash', width=2),
                  annotation_text="Mean", annotation_position="top left", col=col, row=1)
    fig.add_vline(x=median_flipper, line=dict(color='green', dash='dot', width=2),
                  annotation_text="Median", annotation_position="top right", col=col, row=1)
    
    # Add rectangles for ranges
    fig.add_vrect(x0=min_flipper, x1=max_flipper, fillcolor='lightgrey', opacity=0.2, 
                  line_width=0, annotation_text="Range", col=col, row=1)
    fig.add_vrect(x0=q1, x1=q3, fillcolor='yellow', opacity=0.3, line_width=0, 
                  annotation_text="IQR", col=col, row=1)
    fig.add_vrect(x0=mean_flipper - 2 * std_flipper, x1=mean_flipper + 2 * std_flipper, 
                  fillcolor='red', opacity=0.2, line_width=0, annotation_text="±2σ", col=col, row=1)

# Create a facet plot for histograms of each species
fig = px.histogram(penguins, x='flipper_length_mm', color='species', facet_col='species', 
                   title='Penguin Flipper Lengths by Species', marginal="box", 
                   template="plotly_white")

# Loop through each species and add lines and rectangles
for i, species in enumerate(penguins['species'].unique()):
    species_data = penguins[penguins['species'] == species]
    add_lines_rectangles(fig, species, species_data, col=i + 1)

# Show the plot
fig.show()


# In[9]:


#Q2:
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load penguins dataset
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv"
penguins = pd.read_csv(url)

# Remove rows with missing values in key columns
penguins = penguins.dropna(subset=['species', 'flipper_length_mm'])

# Define function to add statistical markers and annotations to KDE plots
def add_statistical_markers(ax, data):
    mean_flipper = data['flipper_length_mm'].mean()
    median_flipper = data['flipper_length_mm'].median()
    std_flipper = data['flipper_length_mm'].std()
    min_flipper = data['flipper_length_mm'].min()
    max_flipper = data['flipper_length_mm'].max()
    q1 = data['flipper_length_mm'].quantile(0.25)
    q3 = data['flipper_length_mm'].quantile(0.75)

    # Add lines for mean and median
    ax.axvline(mean_flipper, color='blue', linestyle='--', label='Mean', linewidth=2)
    ax.axvline(median_flipper, color='green', linestyle=':', label='Median', linewidth=2)

    # Add shaded areas for ranges
    ax.axvspan(min_flipper, max_flipper, color='lightgrey', alpha=0.2, label="Range")
    ax.axvspan(q1, q3, color='yellow', alpha=0.3, label="IQR")
    ax.axvspan(mean_flipper - 2 * std_flipper, mean_flipper + 2 * std_flipper, color='red', alpha=0.2, label="±2σ")

    # Add annotations for mean, median, and standard deviation
    ax.annotate(f'Mean: {mean_flipper:.1f}', xy=(mean_flipper, 0), 
                xytext=(mean_flipper + 2, 0.01), 
                arrowprops=dict(arrowstyle='->', color='blue'), color='blue', fontsize=10)
    
    ax.annotate(f'Median: {median_flipper:.1f}', xy=(median_flipper, 0), 
                xytext=(median_flipper - 3, 0.02), 
                arrowprops=dict(arrowstyle='->', color='green'), color='green', fontsize=10)

    # ±2 Standard Deviation annotation
    ax.annotate(f'±2σ: {mean_flipper:.1f} ± {std_flipper:.1f}', 
                xy=(mean_flipper + 2 * std_flipper, 0), 
                xytext=(mean_flipper + 4, 0.01), 
                arrowprops=dict(arrowstyle='->', color='red'), color='red', fontsize=10)

    # IQR annotation
    ax.annotate(f'IQR: {q1:.1f} to {q3:.1f}', xy=(q3, 0), 
                xytext=(q3 + 2, 0.02), 
                arrowprops=dict(arrowstyle='->', color='yellow'), color='yellow', fontsize=10)
    
    # Range annotation
    ax.annotate(f'Range: {min_flipper:.1f} to {max_flipper:.1f}', 
                xy=(max_flipper, 0), 
                xytext=(max_flipper - 4, 0.02), 
                arrowprops=dict(arrowstyle='->', color='lightgrey'), color='lightgrey', fontsize=10)

# Create subplots: 1 row, 3 columns for each species
fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

# Plot KDEs for each species
species_list = penguins['species'].unique()
for i, species in enumerate(species_list):
    species_data = penguins[penguins['species'] == species]
    
    # Plot KDE for flipper length
    sns.kdeplot(species_data['flipper_length_mm'], ax=axes[i], color='green', fill=True)
    
    # Add title
    axes[i].set_title(f'{species} Flipper Length')
    
    # Add statistical markers
    add_statistical_markers(axes[i], species_data)

# Common labels
fig.suptitle('Penguin Flipper Length Distribution (KDE) by Species', fontsize=16)
plt.tight_layout()
plt.show()


# Q3:Contrasting Descriptions of Data Distribution Visualization Methods
# 
# Box Plots（Description）: Box plots visually summarize data through their quartiles. They show the median, the interquartile range (IQR), and potential outliers. The "box" represents the IQR (25th to 75th percentile), and "whiskers" extend to the minimum and maximum values (excluding outliers).
# 
# Histograms（Description）: Histograms display the frequency distribution of numerical data by dividing the data range into bins (intervals). Each bin's height represents the number of data points that fall within that range. They provide a clear view of the distribution shape and central tendencies.
# 
# Kernel Density Estimators (KDE)（Description）: KDE plots estimate the probability density function of a continuous random variable. Instead of bins, they create a smooth curve by placing a kernel (a smooth, bell-shaped curve) over each data point and summing these curves to create a continuous density estimate. The shape can be influenced by the bandwidth parameter.
# 
# Pros and Cons： Box Plots: Pros:Summarizes key statistics (median, quartiles, outliers) succinctly. Useful for comparing distributions across multiple categories. Clearly identifies outliers. Cons: Does not show the actual distribution of data within the quartiles. Can oversimplify complex distributions.
# 
# Histograms: Pros: Provides a visual representation of data distribution, allowing easy identification of skewness and modality (peaks). Intuitive and easy to interpret. Can reveal patterns in data over various bin sizes. Cons: The choice of bin size can significantly affect the appearance and interpretation. Can obscure underlying distribution if bins are poorly chosen.
# 
# Kernel Density Estimators (KDE): Pros: Provides a smooth estimate of the distribution, revealing features like multi-modality. Less sensitive to the choice of parameters compared to histograms. Visually appealing and useful for comparing distributions. Cons: Bandwidth selection can be subjective and significantly impacts the result. Can mislead if not interpreted carefully (e.g., suggesting false peaks). Personal Preference and Rationale
# 
# I prefer KDE plots for several reasons: KDE plots have a nice, smooth curve that makes it easier to understand the underlying distribution of the data. Unlike histograms, which can look jagged and heavily depend on the choice of bin sizes, KDEs give me a clearer view of the data’s density.
# 
# 
# 

# Summary of communicating with chatgpt(Q1-Q3):
# 
# The link:https://chatgpt.com/share/66f4f23a-6468-800d-b6c2-8532add838ba
# 
# Initial Inquiry on Visualizations:
# 
# You requested guidance on creating visualizations for the flipper_length_mm variable from the penguins dataset, specifically using Plotly to add statistical markers such as mean, median, range, interquartile range (IQR), and standard deviations to histograms.
# Transition to Seaborn KDE Plots:
# 
# After discussing the initial plots, you asked to transition the focus to using Seaborn’s Kernel Density Estimation (KDE) plots for a clearer visual representation of the data, organized in rows of three plots for each species.
# Enhancements for Statistical Markers:
# 
# You requested that the KDE plots include annotations for mean, median, IQR, range, and standard deviations, to provide a comprehensive view of the distribution.
# Visualizations Creation:
# 
# I provided code examples for creating box plots, histograms, and KDE plots using the same penguins dataset, along with explanations for each plot's advantages and limitations.
# User Preferences:
# 
# You shared your preferences for visualizations, expressing that you appreciate KDE plots for their clarity in showing data distribution, while recognizing the strengths of box plots for summarizing data and the potential pitfalls of histograms regarding bin size sensitivity.
# Conclusion
# Throughout our conversation, we explored various data visualization techniques for analyzing penguin flipper lengths, focusing on enhancing clarity and understanding of the data distributions. You expressed a preference for KDE plots for their smooth representation of the data.

# In[12]:


#Q4:
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

n = 1500

data1 = stats.uniform.rvs(0, 10, size=n)  
data2 = stats.norm.rvs(5, 1.5, size=n)    
data3 = np.r_[stats.norm.rvs(2, 0.25, size=int(n/2)), 
               stats.norm.rvs(8, 0.5, size=int(n/2))]  
data4 = stats.norm.rvs(6, 0.5, size=n)    

fig = make_subplots(rows=1, cols=4)


fig.add_trace(go.Histogram(x=data1, name='A', nbinsx=30, 
                             marker=dict(line=dict(color='black', width=1))), row=1, col=1)
fig.add_trace(go.Histogram(x=data2, name='B', nbinsx=15, 
                             marker=dict(line=dict(color='black', width=1))), row=1, col=2)
fig.add_trace(go.Histogram(x=data3, name='C', nbinsx=45, 
                             marker=dict(line=dict(color='black', width=1))), row=1, col=3)
fig.add_trace(go.Histogram(x=data4, name='D', nbinsx=15, 
                             marker=dict(line=dict(color='black', width=1))), row=1, col=4)


fig.update_layout(height=300, width=1200, title_text="Row of Histograms")


fig.update_xaxes(title_text="A", row=1, col=1)
fig.update_xaxes(title_text="B", row=1, col=2)
fig.update_xaxes(title_text="C", row=1, col=3)
fig.update_xaxes(title_text="D", row=1, col=4)


fig.update_xaxes(range=[-0.5, 10.5])


fig.show(renderer="png")  


# #Q4:Answer
# 
# Similar Means and Similar Variances Datasets A and D are quite similar, with both having means around 6 and low variances. This means their data points cluster closely together.
# 
# Similar Means but Quite Different Variances Datasets B and C have means that are pretty close to each other, around 5, but their variances tell a different story. Dataset B has a wider spread, while Dataset C, with its mixture of two normals, has a more peaked distribution but with greater variability.
# 
# Similar Variances but Quite Different Means Datasets A and D have similar variances, which means their data spreads are comparable. However, their means are quite different—A centers around 5 while D is closer to 6.
# 
# Quite Different Means and Quite Different Variances Datasets B and C really stand out from each other. Dataset B has a mean around 5 and a moderate variance, while Dataset C has a different average and a more complex, dual-peaked distribution with higher variance.

# In[13]:


import plotly.express as px
from scipy import stats
import pandas as pd
import numpy as np

sample1 = stats.gamma(a=2, scale=2).rvs(size=1000)
mean1 = sample1.mean()
median1 = np.quantile(sample1, [0.5])

sample2 = -stats.gamma(a=2, scale=2).rvs(size=1000)
mean2 = sample2.mean()
median2 = np.quantile(sample2, [0.5])

df1 = pd.DataFrame({'data': sample1})
df2 = pd.DataFrame({'data': sample2})

fig1 = px.histogram(df1, x="data", title="Right Skewed Distribution")
fig1.show(renderer="png")

fig2 = px.histogram(df2, x="data", title="Left Skewed Distribution")
fig2.show(renderer="png")

print(f"Right Skew - Mean: {mean1}, Median: {median1}")
print(f"Left Skew - Mean: {mean2}, Median: {median2}")


# Q5:
# 
# 1）：Relationship between skewness and mean and median):
# 
# In a right-skewed distribution, the tail on the right side is longer or fatter than the left side. The mean is typically greater than the median because the mean is affected by the higher values in the tail. Left Skew (Negative Skew): In a left-skewed distribution, the tail on the left side is longer or fatter than the right side. The mean is usually less than the median because the mean is pulled down by the lower values in the tail.
# 
# (2):How does the code work:
# 
# Sample Generation: Right Skew: Generates 1,000 random samples from a gamma distribution using stats.gamma(a=2, scale=2). Left Skew: Creates a left-skewed distribution by negating the gamma samples.
# 
# Visualization: Histograms of both right-skewed and left-skewed samples are created using Plotly to visually assess their distributions.
# 
# Calculating Central Tendency: The mean and median of each sample are calculated: For the right-skewed sample, the mean is typically greater than the median. For the left-skewed sample, the mean is usually less than the median.
# 
# Output: Displays the histograms and prints the mean and median for both distributions, highlighting the relationship between skewness and the measures of central tendency.
# 
# (3).Own word summary: In my exploration of the relationship between the mean and median, I found that the skewness of a distribution plays a crucial role in how these two measures interact.
# 
# Right Skewness (Positive Skew) When a distribution is right-skewed, the tail on the right side is longer or fatter than on the left. In this case, I observed that:
# 
# Mean > Median: The mean is higher than the median. This happens because the mean is sensitive to extreme values (or outliers) in the right tail. As a result, a few high values pull the mean upward, while the median, being the middle value, remains more stable.
# 
# 

# Summary of communication with chatgpt:
# The link:https://chatgpt.com/share/66f4f5ce-65f8-800d-b9da-b72b33baee86
# Mean and Median Relationship:
# 
# We discussed how the mean and median relate to skewness in distributions.
# In a right-skewed (positive skew) distribution, the mean is greater than the median due to a longer right tail.
# In a left-skewed (negative skew) distribution, the mean is less than the median due to a longer left tail.
# In a symmetrical distribution, the mean and median are equal.
# User Preferences:
# 
# You expressed a preference for shorter and simpler descriptions, avoiding overly wordy sentences, and a more human-like, first-person perspective in explanations.
# You indicated a preference for KDE plots for visualizing data distribution, while noting the usefulness of box plots for summarizing data and the sensitivity of histograms to bin size.
# 

# In[21]:


#Question6,self-found dataset

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

url = "https://raw.githubusercontent.com/manuelamc14/fast-food-Nutritional-Database/main/Tables/nutrition.csv"
df = pd.read_csv(url)

print(df.head())
print("Column names:")
print(df.columns)

cols_to_convert = ['Total Fat (g)', 'Calories', 'Sodium (mg)', 'Protein (g)']
df[cols_to_convert] = df[cols_to_convert].apply(pd.to_numeric, errors='coerce')

print("\nMissing values in each column:")
print(df.isnull().sum())

summary_stats = df[cols_to_convert].describe()
print("\nSummary statistics:")
print(summary_stats)

plt.figure(figsize=(10, 6))
sns.histplot(df['Calories'].dropna(), bins=30, kde=True)
plt.title('Distribution of Calories in Fast Food Items')
plt.xlabel('Calories')
plt.ylabel('Frequency')
plt.grid()
plt.show()

plt.figure(figsize=(12, 8))
sns.boxplot(data=df, x='Restaurant ID', y='Total Fat (g)')
plt.title('Total Fat Content by Restaurant')
plt.xlabel('Restaurant ID')
plt.ylabel('Total Fat (g)')
plt.xticks(rotation=90)
plt.grid()
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Sodium (mg)', y='Calories', alpha=0.6)
plt.title('Calories vs. Sodium Content in Fast Food')
plt.xlabel('Sodium (mg)')
plt.ylabel('Calories')
plt.grid()
plt.show()


# In[22]:


#Question6 The code about statistics and visualization

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

url = "https://raw.githubusercontent.com/manuelamc14/fast-food-Nutritional-Database/main/Tables/nutrition.csv"
df = pd.read_csv(url)

print(df.head())
print("Column names:")
print(df.columns)

cols_to_convert = ['total fat (g)', 'calories', 'sodium (mg)', 'protein (g)']
df[cols_to_convert] = df[cols_to_convert].apply(pd.to_numeric, errors='coerce')

print("\nMissing values in each column:")
print(df.isnull().sum())

summary_stats = df[cols_to_convert].describe()
print("\nSummary statistics:")
print(summary_stats)

plt.figure(figsize=(10, 6))
sns.histplot(df['calories'].dropna(), bins=30, kde=True)
plt.title('Distribution of Calories in Fast Food Items')
plt.xlabel('Calories')
plt.ylabel('Frequency')
plt.grid()
plt.show()

plt.figure(figsize=(12, 8))
sns.boxplot(data=df, x='restaurant id', y='total fat (g)')
plt.title('Total Fat Content by Restaurant')
plt.xlabel('Restaurant ID')
plt.ylabel('Total Fat (g)')
plt.xticks(rotation=90)
plt.grid()
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='sodium (mg)', y='calories', alpha=0.6)
plt.title('Calories vs. Sodium Content in Fast Food')
plt.xlabel('Sodium (mg)')
plt.ylabel('Calories')
plt.grid()
plt.show()


# Q6 summary:
# Histogram of Calories: Shows the distribution of calorie counts in fast food items.
# Box Plot of Total Fat by Restaurant: Illustrates the range and distribution of total fat content across different restaurants.
# Scatter Plot of Calories vs. Sodium: Displays the relationship between sodium content and calories.

# In[26]:


#Question7
import plotly.express as px
import pandas as pd
import numpy as np

np.random.seed(42)
years = [2000, 2005, 2010, 2015, 2020]
countries = ['Country A', 'Country B', 'Country C', 'Country D']
data = []

for year in years:
    for country in countries:
        gdp = np.random.randint(10000, 50000)
        happiness_score = np.random.uniform(4, 10)
        pop = np.random.randint(1000000, 10000000)
        continent = 'Continent 1' if country in ['Country A', 'Country B'] else 'Continent 2'
        data.append({'country': country, 'gdpPercap': gdp, 'happiness_score': happiness_score, 
                     'year': year, 'pop': pop, 'continent': continent})

df = pd.DataFrame(data)

fig = px.scatter(df, 
                 x="gdpPercap", 
                 y="happiness_score", 
                 animation_frame="year", 
                 animation_group="country",
                 size="pop",           
                 color="continent",    
                 hover_name="country",
                 log_x=True, 
                 size_max=60, 
                 range_x=[1000, 50000], 
                 range_y=[0, 10],
                 color_continuous_scale=px.colors.sequential.Viridis)

fig.update_layout(
    title="GDP vs Happiness Score Over Years",
    xaxis_title="GDP per Capita",
    yaxis_title="Happiness Score",
    legend_title="Continent"
)

fig.update_traces(marker=dict(opacity=0.8, line=dict(width=0.5, color='DarkSlateGrey')))

fig.show()


# In[30]:


#Question8:after some changes
import plotly.express as px
import pandas as pd
import numpy as np

np.random.seed(42)
years = [2000, 2005, 2010, 2015, 2020]
names = ['Name A', 'Name B', 'Name C', 'Name D']
data = []

for year in years:
    for name in names:
        percent_change = np.random.uniform(-0.005, 0.005) 
        rank = np.random.randint(1, 100)  
        percent = np.random.uniform(0, 100)  
        sex = 'Male' if name in ['Name A', 'Name B'] else 'Female' 
        data.append({'name': name, 'percent_change': percent_change, 'rank': rank, 
                     'year': year, 'percent': percent, 'sex': sex})

df = pd.DataFrame(data)


fig = px.scatter(df, 
                 x="percent_change", 
                 y="rank", 
                 animation_frame="year", 
                 animation_group="name",
                 size="percent",           
                 color="sex",    
                 hover_name="name",
                 size_max=50, 
                 range_x=[-0.005, 0.005])

fig.update_layout(
    title="Percent Change vs Rank Over Years",
    xaxis_title="Percent Change",
    yaxis_title="Rank",
    legend_title="Sex"
)

fig.update_traces(marker=dict(opacity=0.8, line=dict(width=0.5, color='DarkSlateGrey')))

fig.show()


# #Question9:
# I find chatgpt very helpful in most of the ways.
# The summary link:
# 
