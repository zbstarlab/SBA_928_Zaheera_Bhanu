SBA 928 - Enhancing Market Research with AI Prompt Engineering

Name: Zaheera Bhanu

Introduction

For this project, I used AI prompt engineering and fine-tuning to analyze customer reviews. My goal was to see if I could train an AI model to identify what customers liked about products, what problems they experienced, and what companies could improve.

Dataset

I used the Women's E-Commerce Clothing Reviews dataset from Kaggle.

Dataset link: https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews

The original dataset contains 23,486 customer reviews. It includes information such as customer age, product category, ratings, and review text.

I initially selected 300 reviews using a random seed of 42. For this experiment, I used 20 annotated reviews.

- Training data: 10 reviews
- Validation data: 5 reviews
- Test data: 5 reviews

I created expected answers for these reviews and used them to train and evaluate the model. I also added validation checks to make sure the required fields were included and review IDs were not duplicated.

Prompt Engineering

I created three different prompts to see how changing the instructions would affect the AI's responses.

The first prompt was minimal and asked the model to analyze a review.

The second prompt was structured and asked for the positive feature, primary problem, and recommendation.

The third prompt was grounded and told the model to use only the information provided in the customer review.

I tested all three prompts on 12 reviews, which gave me 36 responses.

I noticed that the minimal prompt sometimes gave very general answers like "Five Stars." The structured prompt sometimes returned only one word, such as "Positive." The grounded prompt gave more detailed answers in some cases, but it still made mistakes.

Fine-Tuning

I used Google's FLAN-T5-small model for fine-tuning.

I trained the model using the following settings:

- Python version: 3.12.14
- Training examples: 10
- Validation examples: 5
- Training epochs: 3
- Learning rate: 0.0001
- Random seed: 42

The validation loss decreased during training:

- Epoch 1: 2.713
- Epoch 2: 2.516
- Epoch 3: 2.457

This showed that the model was improving at predicting the expected validation answers during training.

Model Comparison

After fine-tuning, I compared the original FLAN-T5-small model with my fine-tuned version using the same five test reviews.

The results were:

- Original model Token F1: 0.056
- Fine-tuned model Token F1: 0.411
- Original model format score: 0%
- Fine-tuned model format score: 60%
- Exact match: 0% for both models

The fine-tuned model performed better at matching the expected answers and following the requested format.

However, it still made mistakes. Sometimes it repeated information, missed recommendations, or generated information that was not mentioned in the review.

Bias Analysis

I checked how the model performed across different age groups, clothing categories, and review ratings.

I found differences in performance:

- Age group Token F1 gap: 0.3711
- Product category Token F1 gap: 0.2491
- Rating group Token F1 gap: 0.3381

Since I only used five test reviews, there was not enough data to determine whether these differences were caused by actual bias.

To improve fairness, I would use more reviews from different age groups and product categories, include more positive and negative reviews, and check model responses manually for inaccurate information.

Custom Review Test

I also tested the fine-tuned model with a new review about a wireless mouse.

The review said that the mouse was comfortable to hold but disconnected several times each day.

The model correctly identified the comfort as a positive feature and the disconnections as the main problem. However, it did not provide a recommendation.

This showed that the model could identify useful information but still needed improvement.

Challenges and Limitations

One challenge was preparing the dataset because the original reviews did not include the expected answers needed for fine-tuning.

Another challenge was that some reviews were incomplete or cut off.

I also used a very small training and testing dataset, so the results are not enough to make strong conclusions about model accuracy or fairness.

Conclusion

This project helped me understand how prompt engineering and fine-tuning work together.

I learned how to prepare a dataset, create different prompts, fine-tune a language model, compare results, and check for possible bias.

The fine-tuned model performed better than the original model on some measurements, but it still made errors.

If I continued this project, I would use more training examples, improve the expected answers, and test the model on a larger and more diverse dataset.

References

Kaggle Dataset:
https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews

Hugging Face:
https://huggingface.co/docs/transformers

PyTorch:
https://pytorch.org/