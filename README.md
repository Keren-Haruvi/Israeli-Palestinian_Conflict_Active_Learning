# Israeli-Palestinian Conflict Active Learning Project

## Project Description

The Israeli-Palestinian conflict is currently a major topic of discussion across the internet. With vast amounts of user-generated content on social media, podcasts, and forums, there is a large, unlabeled dataset that can provide valuable insights into public sentiment. However, manually labeling every comment is impractical due to the massive volume of data. 

Our project aims to develop an active learning-based classifier to efficiently label these comments as either pro-Israeli or anti-Israeli, optimizing the labeling process by focusing on the most informative data points and significantly reducing the manual effort required.

The classifier can help identify platforms or users with a high concentration of anti-Israeli sentiment and highlight the most common anti-Israel arguments, enabling more targeted efforts to present the Israeli perspective. 
Additionally, the project will introduce NLP-specific active learning metrics, with potential applications in other sentiment analysis tasks.

## Team Collaboration and Responsibilities
The virtual machine was connected to Omri's Git account, which is why all commits appear under his name. Throughout the project, we collaborated closely over Zoom, with each team member focusing on specific responsibilities:

- Oren handled dataset collection.
- Omri managed data preprocessing, including augmentation, embeddings, and the initial selection of the training set.
- Keren implemented the active learning pipelines and wrote the report.
- Or implemented the four final models.

Overall, the active learning strategies were developed collaboratively. We all 
contributed to each aspect, discussed everything collectively, and made decisions as a team.

## Prerequisites

Make sure you have the following installed on your system:

- Python 3.11.9 or later
- pip (Python package installer)

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/your-repo-name.git
    cd your-repo-name
    ```

2. Create a virtual environment (optional but recommended):
    ```bash
    conda create --name project_env python=3.11.9
    conda activate project_env
    ```

3. Install the required packages using the `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```

4. Set your OpenAI API key in `Preprocess/data_augmentation.py`:
    ```python
    openai.api_key = "your_api_key"
    ```

## Datasets

In the `Datasets` directory, you will find three DataFrames that comprise the dataset used for this project.

### Data Sources

We collected relevant data from four primary sources:

- **Reddit Posts:**
  The majority of our data was scraped from subreddits such as r/changemyview that specifically discuss the Israeli-Palestinian conflict.
  Find this dataset in `Datasets/reddit_df.csv`

- **Debate Transcriptions:**
  We extracted additional data from transcriptions of debates focused on this topic, capturing both pro-Israeli and anti-Israeli viewpoints.
  Find this dataset in `Datasets/nlp_df.csv`

- **Chain-of-Thought and ChatGPT-Generated Data:**
  We leveraged a carefully crafted prompt to generate synthetic data using ChatGPT.
  We also applied a chain-of-thought reasoning approach to generate additional arguments, utilizing the GitHub repository by [zbambergerNLP/strategic-debate-tot](https://github.com/zbambergerNLP/strategic-debate-tot), expanding the dataset with structured and coherent perspectives on the conflict.
Find this dataset in `Datasets/GPT_df.csv`


## How To Run?

1. **Preprocessing and Data Augmentation:**
   - *Preprocessing Pipeline:*
     To run the preprocessing pipeline, execute:
     ```bash
     python Preprocess/pre_process.py
     ```
     
   - *Generate Embeddings:*
     To create embeddings for the dataset using various NLP models, run:
     ```bash
     python Preprocess/data_augmentation.py
     ```
     
2. **Active Learning:**

   Depending on the active learning strategy you wish to implement, run one of the following:

   - *Single Strategy Active Learning:*
     To run the active learning with a single strategy each time, execute:
      ```bash
      python active_learning.py
      ```

    - *Majority Vote Model:*
      To run the active learning process using the majority vote model, execute:
      ```bash
      python active_learning_majority.py
      ```
    - *Weighted Models:*
      To run the active learning process using weighted models (uniform and adaptive), execute:
      ```bash
      python active_learning_weighted.py
      ```

    - *Weighted + Custom Clusters Model:*
      To run the active learning process using a weighted model combined with custom clusters, execute:
      ```bash
      python active_learning_weighted_custom_kmeans.py
      ```

## Conclusion

Finding the optimal balance between labeling effort and model performance is essential in developing effective algorithms. 
Our analysis indicates that as labeling effort grows, the marginal gains in model performance tend to diminish. 

Using  80 labeled samples, the Adaptive Weight Strategies algorithm achieving an F1 score of 0.846.
This approach leverages strategies that maximize performance while minimizing labeling workload.

## Acknowledgments

- OpenAI for providing the GPT-3.5 API
- Hugging Face for the Transformers library
