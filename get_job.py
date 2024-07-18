import argparse
import os
import datetime
import csv
import time
import requests
from bs4 import BeautifulSoup
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.callbacks import get_openai_callback
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
import tiktoken


def saveRecords(filename, line, header='', replaceCol=-1):
    """
    Save records into a file in this format:
    filename-date.csv

    Args:
        filename (str): Name of the file (without the date suffix).
        line (str): Line to be added to the file.
        header (str, optional): Header of file
        replaceCol (int, optional): Replace column of file if id exists

    Raises:
        AssertionError: If the filename contains a hyphen.
        Exception: If an error occurs while writing to the file.
    """

    # Check if filename has a '-'
    assert '-' not in filename, "Error, filename can't have a ('-') character."

    # Check files in root folder
    root_files = os.listdir('.')

    # Filter filenames with same filename to check if file with same filename exists
    root_files_filtered = list(
        filter(lambda x: x.startswith(filename + '-'), root_files))

    # filename-datetime
    new_filename = filename + '-' + datetime.datetime.now().strftime('%Y%m%d') + '.csv'

    if root_files_filtered:
        # Rename existing file
        os.rename(root_files_filtered[0], new_filename)

    already_exists = False
    job_id = line.split(';')[0]

    try:
        # If file exists, check its content
        if os.path.exists(new_filename):
            with open(new_filename, 'r', encoding='utf-8') as file:
                reader = csv.reader(file, delimiter=';')
                rows = list(reader)
                for row in rows:
                    if row and row[0] == job_id:
                        # If paramenter replaceCol isn't -1 then replace column value in line
                        if replaceCol > -1:
                            row[replaceCol] = line.split(';')[replaceCol]
                        already_exists = True
                        break

        # Write header if the file does not exist and header is provided
        if not already_exists and not os.path.exists(new_filename) and len(header) > 0:
            with open(new_filename, 'a', encoding='utf-8') as file:
                file.write(header + '\n')

        # If replaceCol isn't -1 then write the updated rows back to the file and then just return
        if replaceCol > -1:
            with open(new_filename, 'w', encoding='utf-8', newline='') as file:
                writer = csv.writer(file, delimiter=';')
                writer.writerows(rows)
            return
        # Write line if it isn't in file
        elif not already_exists:
            with open(new_filename, 'a', encoding='utf-8') as file:
                file.write(line + '\n')

        # Verify that the line was written checking the id column
        """
        check_last_line = False
        with open(new_filename, 'r', encoding='utf-8') as file:
            last_line = list(file)[-1].strip().split(';')[0]
            check_last_line = last_line == line.strip().split(';')[0]
            print('id last line: ' + last_line + '\t' +
                  'id line: ' + line.strip().split(';')[0])
        """

    except Exception as e:
        print(f"Error trying to write to file: {new_filename} --- {e}")

    # assert check_last_line, 'Line not written to file or already in file'


def countTokens(text, model='gpt-3.5-turbo'):
    """Calculate tokens of given text using model logic

    Args:
        text (str): Text used to calculate tokens
        model (str, optional): Each model calculates different. Defaults to 'gpt-3.5-turbo'.

    Returns:
        int: Returns number of tokens
    """
    # Get the encoding for the model
    encoding = tiktoken.encoding_for_model(model)

    # Encode the text and get the number of tokens
    tokens = encoding.encode(text)

    return len(tokens)


def recommendJobsOpenAI(contextFile, questionKeyWords, maxRecommendations, tokenLimitRecommendations=0, recursive=0):
    """
    Recommends job listings based on the context file and specified keywords.

    Args:
        contextFile (str): The path to the file containing job descriptions and other relevant context.
        questionKeyWords (list of str): A list of keywords to filter and match jobs against.
        maxRecommendations (int): The maximum number of job recommendations to return.
        tokenLimitRecommendations (int, optional): The maximum number of tokens to use in the recommendation. Defaults to 0 (no limit).
        recursive (int, optional): A flag to enable recursive search for deeper recommendations. Defaults to 0 (disabled).

    Returns:
        list: A list of recommended job descriptions that match the given keywords, up to the maximum specified recommendations.
    """

    # Load key from env
    load_dotenv()

    os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")

    # Check files in root folder
    root_files = os.listdir('.')

    # Filter filenames with same filename to check if file with same filename exists
    root_files_filtered = list(
        filter(lambda x: x.startswith(questionKeyWords.replace(' ', '_') + '-'), root_files))

    questionFile = root_files_filtered[0]

    rows_file = []
    question = ''

    jobsQuantity = 0

    # Get string formatted from csv

    if os.path.exists(questionFile):
        with open(questionFile, 'r', encoding='utf-8') as file:
            reader = csv.reader(file, delimiter=';')
            # Skip header
            next(reader)
            # Get first jobsQuantity that doesn't have recommendations
            for row in reader:
                if row and len(row) == 8 and not row[7].strip():
                    # Here if you put the title your answer will be forced to id-title\int
                    rows_file.append(row)
                    question += row[0] + \
                        '-' """+ row[2] + ', '""" + row[6] + '\n'
                    jobsQuantity += 1
                    if jobsQuantity == maxRecommendations:
                        break
    # All jobs from file are recommended
    if len(rows_file) == 0:
        print('No jobs to recommend')
        return

    # Custom prompt
    template = """Use the following pieces of context to answer the question at the end.
    The question will be in format id-about, for each job answer in this format id-number separated by coma.
    You will put a number for each job, this number must be between 0 and 10 based on the candidate's profile

    Profile: {context}

    Question: {question}

    Your answer:"""

    # Model of preference and temperature 0 because we don't need creative answers
    llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)

    # Get content from our context file
    loader = TextLoader(file_path=contextFile, encoding="utf-8")
    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(data)

    # Total Tokens: The total number of tokens (both input and output) must fit within the context window. For gpt-3.5-turbo, this is 16,385 tokens.
    # Output Tokens: The output alone can be up to 4,096 tokens.

    # Check total tokens for initial question
    token_counter = (countTokens(' '.join(
        [x.text if hasattr(x, 'text') else str(x) for x in splits]) + template + question))

    # Calculate remaining tokens for answer
    remaining_tokens = countTokens('9999999999: 9\n' * jobsQuantity)

    if remaining_tokens > 4096:
        print('Max tokens surprassed for: ' + str(remaining_tokens - 4096))
        # if tokenLimitRecommendations > 0 then call again but with less jobs
        if tokenLimitRecommendations > 0:
            return recommendJobsOpenAI(contextFile, questionKeyWords, maxRecommendations-tokenLimitRecommendations, tokenLimitRecommendations, recursive)
        else:
            return
    elif (token_counter + remaining_tokens) > 16385:
        print('Max tokens surprassed for: ' +
              str(token_counter + remaining_tokens - 16385))
        # if tokenLimitRecommendations > 0 then call again but with less jobs
        if tokenLimitRecommendations > 0:
            return recommendJobsOpenAI(contextFile, questionKeyWords, maxRecommendations-tokenLimitRecommendations, tokenLimitRecommendations, recursive)
        else:
            return

    print('Token estimation: ' + str(token_counter + remaining_tokens))

    # Save the content splitted in the vector store
    vectorstore = FAISS.from_documents(
        splits, OpenAIEmbeddings())

    retriever = vectorstore.as_retriever()

    prompt = PromptTemplate.from_template(template)

    chain = (
        {'context': retriever, 'question': RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    with get_openai_callback() as cb:

        res = chain.invoke(question)
        res = res.split(', ')
        # Parse results
        res = {x.split('-')[0]: x.split('-')[1] for x in res}

        print('Recommendations: ' + str(len(res)))
        print(res)
        # Add result for each row
        for row in rows_file:
            if row and row[0] in res.keys():
                row[7] = (res[row[0]])

            # Write the modified rows back to the CSV file
            saveRecords(filename=questionKeyWords.replace(' ', '_'),
                        line=';'.join(row), replaceCol=7)

        # Check tokens and consumption
        print(cb)
    # If parameter recursive == 1 then use recursion
    if recursive == 1:
        return recommendJobsOpenAI(contextFile, questionKeyWords, maxRecommendations, tokenLimitRecommendations, recursive)


def jobGetterLinkedIn(keyWords, location, last, page, maxJob, quantityJob):
    """Get jobs from LinkedIn and save them in a file, optional send the 

    Args:
        keyWords (string): Search job titles
        location (string): Jobs near location
        last (string): Week or month
        page (string): Page of results, it can go from 0 to quantity/10 to closest 10th (ex: quantity 145 page 15)
        maxJob (int): Max quantity of jobs to get
        quantityJob (int): Used to as registry for maxJob recursion

    Returns:
        _void_: Recursive calling until there isn't any result or quantityJob == maxJob
    """
    # Replaces spaces for api usage
    keyWords = keyWords.replace(' ', '%2B')
    location = location.replace(' ', '%2B')
    last = 'r2592000' if last == 'month' else 'r604800'

    list_url = f"https://www.linkedin.com/jobs/api/seeMoreJobPostings/search?keywords={keyWords}&location={location}&f_TPR={last}&start={page}"
    response = requests.get(list_url)
    # print(str(response))

    # Tries for response when it isn't status 200
    MAX_TRIES = 5
    # Timer in seconds for sleep
    timer_sleep = 3
    # Handle 429 responses
    retries = 0

    while retries < MAX_TRIES:
        if str(response) != '<Response [200]>':
            # Avoid rate limiting, status 429
            time.sleep(timer_sleep)
            response = requests.get(list_url)
            retries += 1
        else:
            break

    if str(response) != '<Response [200]>':
        return print('No more results: ' + str(response))

    # Page will be always like this 0, 10, 20 and so on
    print(f"Page: {page}")

    list_data = response.text
    list_soup = BeautifulSoup(list_data, 'html.parser')
    # Get quantity, for jobs/search
    """quantity = list_soup.find('span', {
        'class': 'results-context-header__job-count'
    }).get_text(' ', True)"""
    # Get every li from page
    page_jobs = list_soup.find_all('li')
    # Save only the first 10 values that aren't empty
    page_jobs = [x for x in page_jobs if x.find(
        'div', {'class': 'base-card'})][:10]

    # Finished if no more jobs found, reached maxJob setted or quantityJob reached 300 jobs
    if (len(page_jobs) == 0) or (quantityJob == maxJob) or (quantityJob >= 300):
        return print('Finished')

    id_list = []

    for job in page_jobs:
        base_card_div = job.find('div', {
                                 'class': 'base-card relative w-full hover:no-underline focus:no-underline base-card--link base-search-card base-search-card--link job-search-card'})
        if base_card_div != None:
            job_id = base_card_div.get('data-entity-urn').split(':')[3]
            if job_id not in id_list:
                id_list.append(job_id)

    result_jobs = []  # dict() for i in range(len(id_list))

    # Avoid rate limiting, status 429
    time.sleep(timer_sleep)

    for job_id in id_list:
        # Handle 429 responses
        retries = 0
        success = False

        while retries < MAX_TRIES and success == False:
            dict_job = {}
            job_url = 'https://www.linkedin.com/jobs-guest/jobs/api/jobPosting/' + job_id
            job_response = requests.get(job_url)
            job_data = job_response.text
            job_soup = BeautifulSoup(job_data, 'html.parser')

            # Get title
            try:
                title = job_soup.find(
                    'h2', {
                        'class': 'top-card-layout__title font-sans text-lg papabear:text-xl font-bold leading-open text-color-text mb-0 topcard__title'
                    }
                ).get_text(' ', True).replace(';', '').replace('\t', ' ').replace('\n', ' ').replace('\r', '')
            except:
                title = None

            # Get company
            try:
                company = job_soup.find(
                    'a', {
                        'class': 'topcard__org-name-link topcard__flavor--black-link'
                    }
                ).get_text(' ', True).replace(';', '').replace('\t', ' ').replace('\n', ' ').replace('\r', '')
            except:
                company = None

            # Get seniority
            try:
                seniority = job_soup.find(
                    'span', {
                        'class': 'description__job-criteria-text description__job-criteria-text--criteria'
                    }
                ).get_text(' ', True).replace(';', '').replace('\t', ' ').replace('\n', ' ').replace('\r', '')
            except:
                seniority = None

            # Get information about the job
            try:
                description = job_soup.find(
                    'div', {
                        'class': 'description__text description__text--rich'}
                ).get_text(' ', True).replace(';', '').replace('\t', ' ').replace('\n', ' ').replace('\r', '')
            except:
                description = None

            # Get date posted
            try:
                date = job_soup.find(
                    'span', {
                        'class': 'posted-time-ago__text'
                    }
                ).get_text(' ', True).replace(';', '').replace('\t', ' ').replace('\n', ' ').replace('\r', '')
            except:
                date = None

            if title and company and seniority and description and date:
                dict_job['id'] = job_id
                dict_job['url'] = job_url
                dict_job['title'] = title
                dict_job['company'] = company
                dict_job['date'] = date
                dict_job['seniority'] = seniority
                dict_job['description'] = description
                result_jobs.append(dict_job)

                # Save values from dict as line separated with ';' and header, one empty space for recommendation
                saveRecords(keyWords.replace('%2B', '_'), ';'.join(str(x) for x in dict_job.values()) + ';',
                            'id;url;title;company;date;seniority;description;recommendation')
                # Exit loop
                success = True
            else:
                # print(str(job_response) + '-' + job_url)
                retries += 1
                if retries == MAX_TRIES:
                    print('Max retries for ' + job_url)
                # Avoid rate limiting, status 429
                time.sleep(timer_sleep)

        # Break loop if maxJob reached
        if len(result_jobs) + quantityJob == maxJob:
            break

        # Avoid rate limiting, status 429
        time.sleep(timer_sleep)
    # Show results
    print('ids: ' + str(len(id_list)) + ' results: ' + str(len(result_jobs)))

    # Change last parameter for recursive calling
    last = 'month' if last == 'r2592000' else 'week'
    # Recursive call the next page, fetching between 10 results
    return jobGetterLinkedIn(keyWords, location, last, str(int(page) + 10), maxJob, quantityJob + len(result_jobs))


# Add parameters to script
if __name__ == '__main__':
    # Stablish parameters for script
    parser = argparse.ArgumentParser()
    parser.add_argument('--key', type=str,
                        help='Key words of job to be searched', required=True)
    parser.add_argument('--location', type=str,
                        help='Country oj job to be searched', required=True)
    parser.add_argument('--last', type=str,
                        help='Since last week or last month', required=True)
    parser.add_argument('--page', type=str,
                        help='Page of results', required=True)
    parser.add_argument('--maxJob', type=int,
                        help='Max quantity of jobs to retrieve', required=True)
    parser.add_argument('--quantityJob', type=int,
                        help='Parameter used to registry total jobs from recursion, initialize with 0', required=True)
    parser.add_argument('--context', type=str,
                        help='Path of context file', required=False)
    parser.add_argument('--maxReco', type=int,
                        help='Number of recommendations', required=False)
    parser.add_argument('--tokenLimit', type=int,
                        help='If not enough tokens then re do the operation but minus this quantity jobs', required=False)
    parser.add_argument('--recursive', type=int,
                        help='Flag to do the operation recursively', required=False)

    args = parser.parse_args()

    jobGetterLinkedIn(keyWords=args.key, location=args.location,
                      last=args.last, page=args.page, maxJob=args.maxJob, quantityJob=args.quantityJob)
    if args.context and args.maxReco:
        recommendJobsOpenAI(contextFile=args.context,
                            questionKeyWords=args.key, maxRecommendations=args.maxReco, tokenLimitRecommendations=args.tokenLimit, recursive=args.recursive)
