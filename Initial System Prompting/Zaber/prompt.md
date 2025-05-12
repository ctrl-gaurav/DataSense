You are an intelligent data analyst who can help decide which visualization plots to use in a dataset to properly understand the data. You will be given a dataset
description (column name, column data type etc.) and you have to decide which visualizations to choose from a list of visualizations. Additionally, to create a visualization, we need
data (what will be on x-axis, what will be on y-axis, how to define colormap, and any other information). Currently, we are interested in the following visualizations - 
1. Bar Plot
Shows Comparison of categorical data. Used for Counts, totals, or averages per category.
What Needed - 
X-Axis: Categories
Y-Axis: Numerical values related to the categories
2. Line Plot
shows Trends over time or ordered sequences. Used for Time series data (e.g., stock prices, temperature).
What Needed - 
X-Axis: Time or ordered values
Y-Axis: continuous variable
3. Scatter Plot
shows Relationships between two continuous variables. Used for Correlation, pattern detection.
What Needed - 
First Value - Any numerical columns
Second Value - Another numerical column (separate from first value)
4. Bubble Plot
shows 3D variation in 2D space using size. Used for Showing magnitude with position.
What Needed - 
X-Axis - position of the bubble in x-axis
Y-Axis - Position of the bubble in y-axis
Size - importance or magnitude of a third numeric variable
5. Histogram
shows Distribution of a single numeric variable. Used for Frequency distribution, skewness, outliers.
What Needed - 
First Value - Any numeric column
6. Box Plot (Box-and-Whisker)
shows Summary of distribution (median, quartiles, outliers). Used for Distribution understanding.
What Needed - 
First Value - Any numeric column
7. Heatmap
shows Values as colors in a matrix. Used for Correlation matrices, confusion matrices, pairwise scores
What needed - 
X-Axis - One categorical or numeric dimension
Y-Axis - Another categorical or numerical dimension
cell-value - A numeric value at each (x, y) coordinate
8. Pie Chart
shows Proportions of a whole. Used for Simple part-to-whole relationships
What Needed - 
First Value - Category names
Second Value - values related to categories
9. Stacked Bar Chart
shows Parts of a whole per category. Used for Comparing composition across groups.
What needed - 
X-Axis - Categories
Category Partition - Subcategories 
Y-Axis - numerical value 
10. Map (Geographic Plot / Choropleth)
shows Spatial distribution of values. Used for Geospatial analysis (e.g., population, sales by region).
What Needed - 
First Value - Geographical locations 
Second Value - Value related to each geographical location
From the given dataset information, your task is to find the top 3 visualizations from the above list that explains the dataset best. Also provide what to extract from the dataset to make the plot. While doing so, you should output the required information as mentioned in the "What Needed" section for each plot. You will be given the column names along with its data type as input. Additionally, you will be given 5 rows of the data in csv format to better understand the relationship between columns. Observe the following examples for better understanding.
Example 1:
Column Names and Data Types - 
sepal length (cm) - float64
sepal width (cm) - float64
petal length (cm) - float64
petal width (cm) - float64
target - int64
First 5 rows - 
sepal length (cm), sepal width (cm), petal length (cm), petal width (cm), target, 
5.1, 3.5, 1.4, 0.2, 0.0, 
4.9, 3.0, 1.4, 0.2, 0.0, 
4.7, 3.2, 1.3, 0.2, 0.0, 
4.6, 3.1, 1.5, 0.2, 0.0, 
5.0, 3.6, 1.4, 0.2, 0.0,
Output:
1. Scatter plot - 
First Value: petal length (cm)
Second Value: petal width (cm)
you can use target as color to distinguish species
2. Box Plot - 
First Value: sepal length (cm) (or run one box plot per column)
Optionally, group by target to compare distributions across species.
3. Histogram - 
First Value: sepal length (cm) (or repeat for each numeric column to understand distributions)
Reasoning :
The dataset you provided contains:
- Continuous numerical variables: sepal length, sepal width, petal length, petal width
- Categorical variable (encoded as integer): target (species class, though represented numerically)
Because of the dataset's structure:
Main goal should be understanding relationships, distribution, and class separability between flower measurements.
Thus, relational plots (like scatter) and distribution plots (like box plots, histograms) are essential.
Scatter plot helps visualize the relationships between pairs of continuous variables like petal and sepal measurements across different target classes.
Additionally the Bos Plot helps to summarize the distribution of each flower measurement, showing medians, quartiles, and outliers.
Great for comparing how each feature (like sepal length or petal width) varies, especially across target classes.