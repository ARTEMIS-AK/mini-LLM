# mini-LLM

## Financial Data Analysis with GPT-2

This project leverages the GPT-2 language model from the Transformers library by Hugging Face to analyze and generate answers based on financial data from a CSV file. The dataset consists of company revenue figures across different periods, and the model is used to generate responses to user queries about this data.

## Project Description

### Installation and Setup
1. **Install Transformers Library**: The `transformers` library is installed using pip.
    ```python
    !pip install transformers
    ```

### Data Loading and Preprocessing
1. **Data Loading**: The dataset is loaded using pandas from a CSV file named `LLM-Sample-Input-File.csv`.
    ```python
    data = pd.read_csv('/content/LLM-Sample-Input-File.csv')
    ```
2. **Data Preprocessing**: The 'Value - Randomized' column is formatted to two decimal places for consistency.
    ```python
    data['Value - Randomized'] = data['Value - Randomized'].apply(lambda x: f'{float(x):.2f}')
    data.head(20)
    ```

### Model Initialization
1. **Tokenization**: The GPT-2 tokenizer is initialized from the pre-trained GPT-2 model.
    ```python
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    ```
2. **Model Loading**: The GPT-2 language model is loaded and moved to the appropriate device (CPU or GPU).
    ```python
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    model.to(device)
    ```

### Query and Response Generation
1. **Function Definition**: A function `generate_answer` is defined to generate answers based on user queries.
    ```python
    def generate_answer(user_question, model):
        input_text = "Question: " + user_question + " Answer:"
        input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)
        output = model.generate(input_ids, max_length=100, num_return_sequences=1, no_repeat_ngram_size=2, early_stopping=True)
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        answer_start = generated_text.find("Answer:") + len("Answer:")
        answer = generated_text[answer_start:].strip()
        return answer
    ```

2. **Interactive Query Session**: An interactive session is set up to continuously accept user queries and provide answers until the user decides to exit.
    ```python
    while True:
        user_query = input("Enter your question (or 'exit' to quit): ")
        if user_query.lower() == 'exit':
            break
        else:
            answer = generate_answer(user_query, model)
            print("Answer:", answer)
    ```

### Example Queries
- "How much revenue does Potato Inc. make from selling Smartphones?"
- "What is the average revenue per user for Smartphone sales?"
- "How much revenue does Potato Inc. make from Japan?"

## Dependencies
- pandas
- torch
- transformers

## Usage
1. Clone the repository.
2. Ensure all dependencies are installed.
3. Place the `LLM-Sample-Input-File.csv` file in the appropriate directory.
4. Run the script to interact with the GPT-2 model and generate responses based on the financial data.

This project demonstrates how to use the GPT-2 language model for natural language understanding and response generation in the context of financial data analysis. It can be further extended to include more complex queries and data processing techniques.
