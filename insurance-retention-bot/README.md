Here is a `README.md` file for your project:

### `README.md`

```markdown
# Insurance Customer Retention Bot

This project is a Streamlit application designed to help insurance companies retain customers at high risk of churn. The bot engages in a conversation with the customers, addresses their concerns, and highlights the benefits of staying with their current policy.

## Folder Structure
```

insurance-retention-bot/
├── app.py
├── gpt_agent.py
├── utils.py
├── config.py
├── data/
│ └── customers.csv
└── requirements.txt

````

### Files and Directories

- **app.py**: Main Streamlit application file that sets up the UI and handles user interactions.
- **gpt_agent.py**: Contains the `GPT` class and related chains (StageAnalyzerChain and ConversationChain).
- **utils.py**: Utility functions for loading customer data and initializing the agent.
- **config.py**: Configuration settings (currently empty but can be used for future configurations).
- **data/**: Directory to store the sample CSV file (`customers.csv`).
- **requirements.txt**: List of dependencies required to run the application.

## Sample CSV File

Ensure your CSV file (`customers.csv`) in the `data/` directory contains the following fields:

| First Name | Last Name | Gender | Age | Region | Occupation | Policy Number | Policy Start Date | Policy Expiry Date | Premium Type | Product Type | Satisfaction Score | Number of Late Payments | Preferred Communication Channel | Number of Customer Service Interactions | Number of Claims Filed | Total Claim Amount | Claim Frequency | Credit Score | Debt-to-Income Ratio |
|------------|------------|--------|-----|--------|------------|---------------|-------------------|--------------------|--------------|--------------|-------------------|------------------------|------------------------------|----------------------------------------|------------------------|------------------|-----------------|-------------------|
| John       | Doe        | Male   | 45  | East   | Engineer   | P123456       | 2019-01-01        | 2024-01-01         | Monthly      | Life         | 80                | 1                      | Email                        | 2                                      | 0                      | 0                | 750             | 0.35              |
| Jane       | Smith      | Female | 50  | West   | Teacher    | P654321       | 2018-06-01        | 2023-06-01         | Quarterly    | Health       | 70                | 3                      | Phone                        | 4                                      | 2                      | 5000             | 0.4              | 720             | 0.30              |

## Setup Instructions

1. **Clone the Repository**:
   ```sh
   git clone <repository_url>
   cd insurance-retention-bot
````

2. **Run Setup Script**:

   ```sh
   bash setup.sh
   ```

3. **Install Dependencies**:

   ```sh
   pip install -r requirements.txt
   ```

4. **Run the Application**:
   ```sh
   streamlit run app.py
   ```

## How It Works

1. **Upload Customer Data**: Upload the `customers.csv` file containing details of customers at high risk of churn.
2. **Start Conversation**: Click the "Start Conversation" button to begin the interaction with a selected customer.
3. **Progress Through Stages**: The bot will progress through predefined conversation stages, addressing the customer's concerns and highlighting policy benefits.

## Key Components

### app.py

The main entry point of the application. It sets up the Streamlit interface and handles user interactions.

### gpt_agent.py

Contains the `GPT` class, which manages the dialogue flow and interactions. It includes:

- **StageAnalyzerChain**: Determines the current stage of the conversation based on history.
- **ConversationChain**: Generates appropriate responses based on the conversation stage and history.

### utils.py

Utility functions for loading customer data and initializing the GPT agent.

### config.py

Configuration settings for the application (currently empty but can be extended).

## Sample Data

A sample `customers.csv` file is provided in the `data/` directory to demonstrate the required data structure and fields.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For questions or support, please open an issue in the repository.

```

This `README.md` file provides a comprehensive overview of the project, including setup instructions, explanations of the key components, and a sample CSV file structure. It is formatted with proper markdown to highlight important sections and improve readability.
```
