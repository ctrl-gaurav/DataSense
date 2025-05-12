We are building a system that automatically generates visualizations (e.g., bar charts, scatter plots, line graphs) from a dataset. To evaluate the quality and correctness of these plots, we ask simple yet targeted questions that can be answered just by looking at the plot.

You are a visualization assistant. Your task is to generate three simple but effective questions based on the following- 

## A dataset description 
    - column names + data types 
    - summary statistics like mean, min, max, and standard deviation of numeric columns
    - A few sample rows

## A plot description
    - plot type, i.e, bar plot/scatter plot, etc
    - columns used 
    - any transformations, if used

These questions should be:
- Answerable by looking directly at the plot, without needing to read the raw data.
- Cover diverse aspects such as trends, patterns, outliers, comparisons, and distributions.
- Designed to help a human quickly verify whether the plot is correctly constructed and faithfully represents the dataset.
- Encourage interpretation of relationships, trends, deviations, and underlying causes.
- Help evaluate whether the plot allows clear and meaningful analytical insight.
- Focus on catching issues such as mislabeled axes, wrong chart types, mismatched data trends
- Be usable for user story. For example, a user might be interested in a comparison between different data types (like a difference between two cities or value differences between two classes)

Think of these as surface-level but sharp diagnostic questions.

While generating questions, avoid the following:
- Deep inference or domain-specific interpretation
- Vague prompts like “What does the plot show?”








### Example 1:

## Dataset Description:
	Column Names and Data Types - 
		- Region (string)
        - Product Category (string)
        - Units Sold (integer)
        - Unit Price (float)
        - Total Revenue (float)
        - Customer Satisfaction Score (float, 0–10 scale)
    Summary Statistics:
        Units Sold: mean = 220, std = 60, min = 100, max = 350
        Unit Price: mean = $25.5, std = 7.1, min = $10, max = $40
        Total Revenue: mean = $5,500, std = 1,800 
        Customer Satisfaction Score: mean = 7.8, std = 1.1, min = 5.0, max = 9.8
        Region: 4 unique values – North, South, East, West
        Product Category: 3 unique values – Electronics, Furniture, Stationery
    Sample Rows:
		Region,Product Category,Units Sold,Unit Price,Total Revenue,Customer Satisfaction Score
        North,Electronics,300,30,9000,8.2
        South,Furniture,180,25,4500,7.5
        East,Stationery,210,15,3150,7.9
        West,Electronics,150,40,6000,6.8
        North,Furniture,250,22,5500,8.5

### Plot Description:
    Type: Bar plot
    X-axis: Product Category
    Y-axis: Average Total Revenue
    Grouping: Bars are grouped by Region (e.g., cluster bars for Electronics by North/South/East/West)

Generated Questions
- Which regions for electronics have higher revenue compared to the West region 
(Reasoning: This helps check the sanity of plot. This information can be easily verified using the dataset.)
- Which region is most suitable for Furniture.
(Reasoning: Help with the user story. A user might need this information to start a business.)
- Which Region provides the best revenue across all product categories?
(Reasoning: Helps with user story)

While providing questions for original inputs, you must not output any reasoning. Only output the questions, nothing more.
