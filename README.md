SBA 928 - Enhancing Market Research with AI Prompt Engineering

Name: Zaheera Bhanu

Project Description

For this project, I used Python and AI to analyze customer reviews and find useful market research information.

I used the Women's E-Commerce Clothing Reviews dataset from Kaggle and Google's FLAN-T5-small language model.

The goal was to identify positive product features, customer complaints, and recommendations for improvement.

Dataset

Dataset link:
https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews

The original dataset contains 23,486 reviews.

I used 20 annotated reviews for the experiment:

- 10 training reviews
- 5 validation reviews
- 5 test reviews

Tools Used

- Python 3.12
- uv
- PyTorch
- Transformers
- Pandas
- NumPy
- Pytest
- Ruff
- Visual Studio Code

How to Run the Project

First, install the required packages:

`uv sync --group dev`

Step 1 - Generate the dataset:

`uv run python -m sba_928_zaheera_bhanu.step_01_generate_dataset`

Step 2 - Test different prompts:

`uv run python -m sba_928_zaheera_bhanu.step_02_prompt_engineering --limit 12`

Step 3 - Fine-tune the model:

`uv run python -m sba_928_zaheera_bhanu.step_03_fine_tune --config config/default.json`

Step 4 - Compare the original and fine-tuned models:

`uv run python -m sba_928_zaheera_bhanu.step_04_compare_models --fine-tuned-model artifacts/models/flan-t5-sba-928/final`

Step 5 - Run the bias analysis:

`uv run python -m sba_928_zaheera_bhanu.step_05_bias_audit`

Step 6 - Test a new customer review:

`uv run python -m sba_928_zaheera_bhanu.step_06_custom_inference --model artifacts/models/flan-t5-sba-928/final --product "Wireless Mouse" --rating 3 --review "The mouse is comfortable but disconnects frequently."`

Testing

To run the tests:

`uv run python -m pytest tests/ -v`

To check the code:

`uv run ruff check .`

Results

After fine-tuning, the model's Token F1 score improved from 0.056 to 0.411.

The format score improved from 0% to 60%.

The model still made mistakes, especially when generating recommendations.

The full results and bias analysis are included in the docs/report.md file.

Limitations

I only used 20 annotated reviews, so this was a small experiment. A larger dataset would be needed to get more reliable results.

The trained model checkpoint can be recreated by running Step 3 if it is not included in the repository.

Dataset source:
https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews