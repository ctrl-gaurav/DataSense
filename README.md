# 📊 DataSense: An Intelligent Data Visualization and Story Generator

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-streamlit-app-url.com) **Tired of manually exploring data and struggling to find the right visualization? Let DataSense do the heavy lifting!**

DataSense leverages the power of Large Language Models (LLMs) to automatically analyze your datasets, recommend the most insightful visualizations, and generate compelling data stories. Go from raw data to beautiful, understandable visuals and narratives in just a few clicks.

![DataSense Screenshot Placeholder](https://placehold.co/800x400/E2E8F0/4A5568?text=DataSense+App+Screenshot)
*(Replace the placeholder above with an actual screenshot of your running application)*

---

## ✨ Features

* **Multi-Format Data Input:** Upload your data as CSV, JSON, or TXT files, or choose from built-in sample datasets (powered by scikit-learn).
* **Automated Data Analysis:** DataSense automatically preprocesses your data (handling types, cleaning names) and performs initial statistical analysis.
* **LLM-Powered Insights:** Utilizes state-of-the-art LLMs to understand the nuances of your data and identify key patterns, correlations, and distributions.
* **Ensemble Recommendations:** Employs a configurable ensemble of multiple LLMs (e.g., Llama 3.1, Qwen 2.5, Phi-3) with majority voting (pass@k sampling) for diverse perspectives and robust visualization suggestions.
* **Top Visualization Suggestions:** Recommends not just one, but the top 1-3 most suitable visualization types based on LLM consensus.
* **Engaging Data Storytelling:** Generates clear, narrative data stories that explain the insights revealed by the recommended visualization, making complex data understandable.
* **Interactive Visualizations:** Creates beautiful, interactive plots using Plotly Express and Plotly Graph Objects.
* **Manual Explorer:** Allows users to manually select and generate any suitable visualization type identified during the analysis phase.
* **Configurable LLM Backend:**
    * Choose which LLMs to include in the ensemble via `config.yaml`.
    * Select loading strategy: `sequential` (less VRAM) or `concurrent` (faster, more VRAM).
    * Configure GPU utilization.
* **User-Friendly Web Interface:** Built with Streamlit for an intuitive and interactive user experience, including progress indicators during analysis.

---

## 🚀 Technology Stack

* **Backend:** Python 3.9+
* **LLM Inference:** [vLLM](https://github.com/vllm-project/vllm)
* **LLM Models:** [Hugging Face Transformers](https://huggingface.co/docs/transformers/index) (for tokenizers, supports various models)
* **Web Framework:** [Streamlit](https://streamlit.io/)
* **Data Handling:** [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
* **Visualization:** [Plotly](https://plotly.com/python/)
* **Sample Datasets:** [Scikit-learn](https://scikit-learn.org/stable/)
* **Configuration:** PyYAML

---

## ⚙️ Setup and Installation

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/your-username/datasense.git](https://github.com/your-username/datasense.git) # Replace with your repo URL
    cd datasense
    ```

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    * **Important:** `vLLM` installation can be complex and depends heavily on your CUDA version and GPU hardware. Please refer to the [official vLLM installation guide](https://docs.vllm.ai/en/latest/getting_started/installation.html) if you encounter issues. Ensure your PyTorch version matches your CUDA toolkit.

4.  **Configure DataSense:**
    * Edit the `config.yaml` file.
    * **`llm_settings`**:
        * `models_to_use`: List the Hugging Face model IDs you want in the ensemble (ensure they are compatible with vLLM and accessible).
        * `loading_strategy`: Choose `'sequential'` (loads models one by one, less VRAM) or `'concurrent'` (loads all models at once, faster, requires high VRAM).
        * `gpu_memory_utilization_per_model`: Adjust GPU memory fraction per model (especially relevant for `concurrent` mode).
        * `num_gpus`: Set the number of GPUs vLLM should use (`tensor_parallel_size`).
        * `sampling_params`: Configure LLM generation parameters like `temperature`, `top_p`, `max_tokens`, and `n` (number of samples per model for pass@k).
    * **`visualization_settings` / `data_settings`**: Adjust defaults if needed.

---

## ▶️ Running the Application

Once the setup is complete, run the Streamlit application from the project's root directory:

```bash
streamlit run app.py
Open the URL provided by Streamlit (usually http://localhost:8501) in your web browser.📖 UsageSelect Data Source: Use the sidebar to either "Upload File" (CSV, JSON, TXT) or select a "Sample Dataset".Preview Data: A preview of the loaded data will appear on the main page.Analyze: Click the "✨ Analyze and Visualize" button in the sidebar.View Results: DataSense will:Show processing status and progress.Display the Top 1-3 recommended visualization types.Show the interactive plot for the primary recommendation.Present the generated "Data Insights" and "Data Story" below the plot.Offer a "Manual Visualization Explorer" to generate other suitable plots.Provide expanders for LLM voting details, metadata, and considered options.🔧 Configuration (config.yaml)The config.yaml file controls key aspects of DataSense:llm_settings: Defines which LLMs to use, how they are loaded (sequentially or concurrently), GPU allocation, and generation parameters (temperature, top_p, max tokens, sampling count n).visualization_settings: Placeholder for future visualization-specific settings.data_settings: Controls data analysis parameters like correlation thresholds, preview sizes, and categorical detection heuristics.🤝 ContributingContributions are welcome! If you'd like to contribute, please feel free to fork the repository, make your changes, and submit a pull request. For major changes, please open an issue first to discuss what you would like to change.(Optional: Add more specific contribution guidelines, e.g., coding standards, testing procedures)🧑‍💻 Team / AuthorsGaurav SrivastavaAafiya HussainNajibul SarkerZaber Hakim📜 CitationIf you use DataSense in your research or work, please cite it as follows:Plain Text:Srivastava, G., Hussain, A., Sarker, N., & Hakim, Z. (2025). DataSense: An Intelligent Data Visualization and Story Generator. [Software]. Available from [https://github.com/your-username/datasense](https://github.com/your-username/datasense)
BibTeX:@software{DataSense2025,
  author = {Srivastava, Gaurav and Hussain, Aafiya and Sarker, Najibul and Hakim, Zaber},
  title = {{DataSense: An Intelligent Data Visualization and Story Generator}},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  url = {[https://github.com/your-username/datasense](https://github.com/your-username/datasense)} # Replace with your actual repo URL
}
(Remember to replace the placeholder URL with your actual repository URL)📄 LicenseThis project is licensed under the MIT License -