# Task Objective:

The goal of the DataSense project is to automate the process of generating visualizations based on a dataset provided by the user. The system should suggest appropriate visualizations based on the data structure and help users gain valuable insights without requiring technical expertise.

We need you to analyze the metadata of the dataset (such as column types and descriptions) and suggest visualizations that would best serve the user's needs. For each visualization, you should provide a rationale and specify which columns should be mapped to the visualization's inputs.

# Instructions:
Dataset & Metadata: You will be provided with a dynamic set of metadata that describes the columns in the dataset. This includes column names, data types (e.g., categorical, numerical, date/time), and a brief description of each column's content, among other metadata.

Choose from the following visualization types: Below are various types of visualizations that could apply to different datasets. You should choose the visualization(s) that best suit the dataset’s characteristics and the insights the user would likely seek.

## Visualization Options
Below are the visualizations you should choose from. When selecting a visualization, ensure that your rationale explains why it is the best choice for the dataset and user goals.

# Bar Chart (Vertical / Horizontal)
Input Parameter Mapping:

x: Categorical column (e.g., Category, Region)

height or width: Numerical column (e.g., Sales, Amount)

Rationale: Ideal for comparing the frequency or total values across different categories.

###  Pie Chart
Input Parameter Mapping:

x: Categorical column (e.g., Category, Region)

sizes: Numerical column representing proportions (e.g., Sales, Count)

Rationale: Displays the proportion of categories as slices of a circle, useful for understanding relative contributions.

### Line Chart
Input Parameter Mapping:

x: Date/Time column (e.g., Date, Timestamp)

y: Numerical column (e.g., Sales, Revenue)

Rationale: Tracks the change in a numerical variable over time, ideal for showing trends or fluctuations.

#  Histogram
Input Parameter Mapping:

x: Continuous numerical column (e.g., Age, Income, Price)

bins: Number of bins or range for the histogram

Rationale: Useful for showing the distribution of a continuous variable, such as the frequency of age or income.

###  Box Plot (Box-and-Whisker Plot)
Input Parameter Mapping:

x: Categorical column (e.g., Category, Region)

y: Continuous numerical column (e.g., Sales, Amount)

Rationale: Shows the distribution, median, quartiles, and outliers of numerical data across categories.

###  Scatter Plot
Input Parameter Mapping:

x: Continuous numerical column (e.g., Age, Advertising Spend)

y: Continuous numerical column (e.g., Sales, Revenue)

Rationale: Displays the relationship or correlation between two numerical variables.

###  Heatmap
Input Parameter Mapping:

data: Correlation matrix or a 2D array (e.g., pairwise correlations between numerical columns)

cmap: Color map for visualization (e.g., 'coolwarm', 'viridis')

Rationale: Visualizes the intensity of correlations or relationships between multiple variables.

###  Stacked Bar Chart
Input Parameter Mapping:

x: Categorical column (e.g., Region, Category)

height: Numerical column (e.g., Sales)

bottom: Stacked parts of the bars (e.g., sub-categories within the main category)

Rationale: Compares part-to-whole relationships across multiple categories, showing how individual segments contribute to the total.

###  Area Chart
Input Parameter Mapping:

x: Date/Time column (e.g., Date, Month)

y: Continuous numerical column (e.g., Cumulative Sales, Cumulative Revenue)

Rationale: Similar to a line chart, but emphasizes the cumulative magnitude of data over time.

###  Treemap
Input Parameter Mapping:

labels: Categorical column (e.g., Product Category)

sizes: Numerical column (e.g., Sales, Revenue)

colors: Categorical or numerical column (optional for coloring)

Rationale: Displays hierarchical data, showing the size of each category within a compact and visually appealing structure.

###  Violin Plot
Input Parameter Mapping:

x: Categorical column (e.g., Product Category, Region)

y: Continuous numerical column (e.g., Sales, Price)

Rationale: Compares the distribution of numerical data between multiple categories, combining aspects of a box plot and a density plot.

###  Pair Plot
Input Parameter Mapping:

data: Dataframe containing multiple continuous columns (e.g., Sales, Profit, Revenue)

hue: Categorical column (optional for coloring by categories)

Rationale: Displays pairwise relationships between multiple numerical variables, useful for detecting correlations.

###  Radar Chart (Spider Chart)
Input Parameter Mapping:

labels: List of categories (e.g., Product Features)

values: List of numerical values corresponding to each category (e.g., Customer Ratings)

Rationale: Compares multivariate data across multiple categories in a circular layout, useful for performance metrics.

###  Density Plot
Input Parameter Mapping:

x: Continuous numerical column (e.g., Age, Income)

bw_method: Bandwidth method for smoothing

Rationale: Displays the distribution of a continuous variable, highlighting the density of data points across different values.

### Stacked Area Chart
Input Parameter Mapping:

x: Date/Time column (e.g., Date, Month)

y: Numerical column for each category (e.g., Sales by Region)

stacked: Boolean to stack areas on top of each other

Rationale: Similar to an area chart but shows the contribution of multiple categories over time.

###  Contour Plot
Input Parameter Mapping:

x: Continuous numerical column (e.g., Latitude)

y: Continuous numerical column (e.g., Longitude)

z: Numerical column for contour levels (e.g., Temperature)

Rationale: Used to represent three-dimensional data in two dimensions, showing levels of a continuous variable across a grid.

###  Bubble Chart
Input Parameter Mapping:

x: Continuous numerical column (e.g., Advertising Spend)

y: Continuous numerical column (e.g., Sales)

size: Continuous numerical column (e.g., Customer Count)

color: Categorical or numerical column for coloring the bubbles

Rationale: Adds a third dimension (size) to a scatter plot, visualizing the relationship between three variables.

### Map (Geographic Plot / Choropleth)
Input Parameter Mapping:

First Value - Geographical locations 

Second Value - Value related to each geographical location

Rationale: Shows Spatial distribution of values. Used for Geospatial analysis (e.g., population, sales by region).

###  Streamgraph
Input Parameter Mapping:

data: Data for time-series categories

x: Date/Time column (e.g., Date, Year)

y: Numerical value for each time point (e.g., Sales)

Rationale: Displays changes in multiple categories over time in a stacked, flowing layout, emphasizing trends and transitions.

###  Word Cloud
Input Parameter Mapping:

text: Textual data (e.g., customer feedback, reviews, etc.)

width, height: Dimensions of the image (optional)

max_words: Maximum number of words to display (optional)

Rationale: Visualizes the frequency of words in a textual dataset, highlighting the most common terms.


# Example Output:
### Example 1: Output for a Dataset with Sales Data
1. Bar Chart (Vertical)
Rationale: A bar chart is an excellent choice for comparing the total sales by product category. It will help the user quickly assess which product categories are performing better than others. This type of visualization is particularly useful when comparing discrete categories such as product types or regions.

Column Mapping:

x: Product Category (Categorical column describing product types)

height: Sales (Numerical column representing the total sales for each product category)

2. Line Chart
Rationale: A line chart is the best option for visualizing sales trends over time. It helps the user understand how sales change over a period, which is crucial for identifying seasonal trends, growth patterns, or any fluctuations. The Date column will be used for the x-axis, and the Sales column will represent the sales values over time.

Column Mapping:

x: Date (Date/Time column showing the date of each sale)

y: Sales (Numerical column representing the total sales on that date)

3. Box Plot
Rationale: A box plot is useful for understanding the distribution of sales within each product category. It shows the median, range, quartiles, and any outliers, providing insights into the consistency of sales across different product categories. It can highlight any irregular patterns or extreme sales values.

Column Mapping:

x: Product Category (Categorical column describing product types)

y: Sales (Numerical column representing sales values)

### Example 2: Output for a Dataset with Customer Information
1. Scatter Plot
Rationale: A scatter plot is appropriate for visualizing the relationship between customer age and income. It allows the user to assess if there is any correlation between these two variables. For example, it could show if older customers tend to have higher incomes or if income is more evenly distributed across different age groups.

Column Mapping:

x: Age (Continuous numerical column representing the customer's age)

y: Income (Continuous numerical column representing the customer's income)

2. Heatmap
Rationale: A heatmap is a great way to visualize correlations between multiple continuous variables. In this case, a correlation matrix of Age, Income, and Spending can be displayed to help the user understand the relationships between these numerical variables. The heatmap will show which variables have strong positive or negative correlations with each other.

Column Mapping:

data: Correlation matrix between Age, Income, and Spending (Numerical columns representing customer attributes)

cmap: 'coolwarm' (Optional color map to visually differentiate correlation strength)

3. Violin Plot
Rationale: A violin plot is useful for comparing the distribution of a continuous variable (such as income) across categories (e.g., gender, product type). It combines aspects of both box plots and kernel density plots, showing the spread and density of the data. In this case, it can be used to compare the income distribution between different customer types (e.g., gender or product preference).

Column Mapping:

x: Gender (Categorical column describing the gender of customers)

y: Income (Continuous numerical column representing income)



# Final Instructions:
1. Choose Visualizations Carefully: Select visualizations that best match the dataset’s characteristics and user goals. For instance, use line charts for time-based data, bar charts for categorical comparisons, and scatter plots for numerical relationships. Remember that the visualizations shouldn't require the dataset to be changed in anyway (you cannot modify the data for input in the chosen visualizations)

2. Column-to-Input Mappings: Ensure that the column names are mapped correctly to each visualization’s input fields (e.g., X-axis, Y-axis).

3. Provide Clear Rationale: Justify each visualization choice in terms of the insights it provides, explaining how the visualization will help the user understand the dataset’s structure.

# User

## Dataset & Metadata: 
{PLACEHOLDER_DATASET_METADATA}

## Output:
Given the above, now provide the output according to the instructions. Limit yourself to the BEST visualization options (shouldn't be more than 3 options). Try to cover breadth and depth of visualization insights.