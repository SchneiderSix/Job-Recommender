# Job Recommender Script

This script automates the process of fetching job listings from LinkedIn, saving them into a CSV file, and recommending jobs based on a given context using RAG method with model OpenAI's GPT-3.5-turbo.

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Setup](#setup)
- [Usage](#usage)
- [Parameters](#parameters)
- [Example](#example)
- [Considerations](#considerations)
- [License](#license)
- [Contact](#contact)

## Features

1. **Job Fetching**: Scrapes job listings from LinkedIn based on specified keywords, location, and date range.
2. **CSV Storage**: Saves the fetched job listings into a CSV file with a date suffix.
3. **Job Recommendation**: Uses a context file (e.g., a candidate's profile) to generate job recommendations with scores between 0 and 10.
4. **Recursive Fetching and Recommendation**: Supports recursive fetching and recommendation to handle large datasets and token limits.

## Requirements

- Poetry
- Python 3.6+
- Required libraries (poetry will handle dependencies):
  - argparse
  - os
  - datetime
  - csv
  - time
  - requests
  - BeautifulSoup
  - langchain_community
  - langchain_core
  - langchain_openai
  - dotenv
  - tiktoken

## Setup

1. **Install required libraries**:

   ```bash
   poetry install
   ```

2. **Set up environment variables**:
   - Create a `.env` file in the root directory of the script.
   - Add your OpenAI API key:
     ```env
     OPENAI_API_KEY=your_openai_api_key
     ```

## Usage

```
poetry run python .\get_job.py --key "<keywords>" --location "<location>" --last "<week|month>" --page "<page>" --maxJob <max_jobs> --quantityJob <quantity_jobs> [--context "<context_file_path>"] [--maxReco <max_recommendations>] [--tokenLimit <token_limit>] [--recursive <recursive_flag>]
```

## Parameters

- `--key` (required): Keywords for job search.
- `--location` (required): Job location.
- `--last` (required): Time range for job posting (`week` or `month`).
- `--page` (required): Starting page for job search results.
- `--maxJob` (required): Maximum number of jobs to retrieve.
- `--quantityJob` (required): Initial job count (set to 0 for the first run).
- `--context` (optional): Path to the context file for recommendations.
- `--maxReco` (optional): Maximum number of job recommendations.
- `--tokenLimit` (optional): Token limit for recommendations.
- `--recursive` (optional): Enable recursive fetching and recommendation (1 to enable).

## Example

1. **Command:**

   ```bash
   poetry run python .\get_job.py --key 'DevOps Developer' --location Uruguay --last week --page 0 --maxJob 30 --quantityJob 0 --context context.txt --maxReco 30 --tokenLimit 10 --recursive 1
   ```

2. **Result:**
   - File Generated: [DevOps_Developer-20240718](DevOps_Developer-20240718.csv)
   - You can change the header or the structure of file [here](get_job.py#L416)

<br />

## Considerations

#### Automated Job Applications

If you plan to automate the job application process, here are some key points to consider:

1. **Easy Apply:**
   If the job listing on LinkedIn has an 'Easy Apply' option, you can automate the application process. Ensure you have an updated pool of possible inputs and answers for the application form using web scraping.
   <br />

2. **External Applications:**
   If the job requires completing the application on an external site, the process becomes more complex. You will need to scrape the external site, understand its HTML structure, and automate the filling of each form element. This is a more involved process but still possible with thorough analysis and coding.
   <br />

3. **Rate Limiting and Bans:**
   When automating applications, especially on LinkedIn, be mindful of rate limiting and account bans. Introduce delays between actions to avoid detection. It's recommended to wait a few seconds between each action to simulate human behavior.
   <br />

4. **Web Scraping Flexibility:**
   You can perform web scraping on any site as long as you understand its HTML structure. This flexibility allows you to extend the script to other job boards and company websites.
   <br />

#### Data Storage

While this script saves results to a CSV file for simplicity, you can store the results in any data storage system you prefer. A non-relational key-value database is recommended for scalability and performance. However, CSV is chosen here for its accessibility and ease of understanding for a broader audience.

#### Recommendation via RAG

This script uses Retrieval-Augmented Generation (RAG) to provide job recommendations. To use this feature, you need a context file that provides a candidate profile. The script uses this profile in a prompt to generate job recommendations. The provided template can be improved to better match your needs.

The script calculates the number of tokens for both the question and the answer. If the token limit is surpassed, a message is displayed. You can choose to use recursion to handle fewer job offers in each iteration if this happens. At the end of execution, the script will show the results, including the cost of the API call.

## License

This project is licensed under the MIT License.

<br />

## Contact

Ask me anything :smiley:

[Juan Matias Rossi](https://www.linkedin.com/in/jmrossi6/)
