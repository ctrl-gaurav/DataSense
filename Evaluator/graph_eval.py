from PIL import Image
from google import genai

from utils import get_gemini_client, replace_placeholder_in_md

def generate_answers(client, chart_path, dataset_description, plot_description, questions):
    print("Entering STAGE 2 of evaluator: generate answers from plot")
    # setup prompt
    prompt = replace_placeholder_in_md(
        "eval_prompt.md",
        {
            "{PLOT DESCRIPTION}" : plot_description,
            "{QUESTION_LIST}" : questions
        }
    )

    print("Using the following prompt:")
    print(prompt)

    # Call the Gemini API to generate answers
    image = Image.open(chart_path)
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[image, prompt]
    )
    print("Response from Gemini API:")
    print(response.text)
    return response.text


def generate_evaluation_score(client, question_answers, dataset_description, plot_description):
    print("Entering STAGE 3 of evaluator: generate evaluation score")
    # setup prompt
    prompt = replace_placeholder_in_md(
        "eval_verifier_prompt.md",
        {
            "{DATASET DESCRIPTION}" : dataset_description,
            "{PLOT DESCRIPTION}" : plot_description,
            "{QUESTION ANSWER}" : question_answers,
        }
    )
    print("Using the following prompt:")
    print(prompt)

    # Call the Gemini API to generate evaluation score
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[prompt]
    )
    print("Response from Gemini API:")
    print(response.text)
    return response.text

if __name__ == "__main__":
    # Load the Gemini client
    client = get_gemini_client()


    ############### GET THESE VALUES FROM EVALUATION PIPELINE 1 ###############

    ### Sample Input Values:

    ### CHART
    chart_path = "iris project.png"


    ### Dataset Description
    dataset_description = """
    Dataset Description:
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
    """


    ### Input Plot Description
    plot_description = """
    Type: Scatter plot

    X-axis: petal length (cm)

    Y-axis: petal width (cm)

    Color Encoding: Points are color-coded by target (Iris class label)

    0: Setosa (e.g., blue)

    1: Versicolor (e.g., green)

    2: Virginica (e.g., red)

    Title: “Petal Length vs. Petal Width by Iris Species”

    Legend: Species labels are shown to help interpret the color-coded classes
    """


    questions = """
    1. Which Iris species forms the tight cluster of points with petal lengths under 2 cm, and what is the approximate range of petal widths for that cluster?

    2. Which species reaches the highest petal width values on the plot, and around what petal length do those points occur?

    3. Which species shows the greatest vertical spread (variation) in petal width values, and how does that spread compare to its horizontal spread (variation) in petal length?
    """

    
    ####### EVALUATOR STEP 2: Generate question answers from plot #######
    answers = generate_answers(
        client,
        chart_path,
        dataset_description,
        plot_description,
        questions
    )

    ####### EVALUATOR STEP 3: Generate evaluation score #######
    score = generate_evaluation_score(
        client,
        answers,
        dataset_description,
        plot_description
    )

    print("Generated score")
    try:
        print(score.split("SCORE")[1])
    except:
        print("Error parsing score from response.")
        print(score)


