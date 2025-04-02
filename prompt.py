import re
from openai_utils import get_completion

def plan_generation(query, old_plan, model):
    prompt_system = "You are an expert plan generation assistant."
    prompt_user = f"""You are given a question over a table. Your task is to generate a plan by decomposing the question into a series of clear, non-overlapping sub-questions, so that answering each sub-question in order will allow you to answer the original question comprehensively. If no decomposition is needed, just return the original question.

If an old plan is provided, you must:
- Carefully review the old plan.
- Identify any missing aspects, vague steps, redundancies, or logic gaps.
- Revise and improve the plan, making it more complete, clear, and logically ordered.
- Ensure the new plan covers all key elements of the original question, fixes old plan issues, and is at least as good as, and preferably better than, the old plan.
- If the old plan is already perfect, you may keep it, but strive for any possible improvement.

**Output only the improved plan in a numbered list (1. 2. 3. ...), without explanations and any other words above or below the plan.**

Examples:

Example 1:
Question: Which country has won the OGAE Video Contest multiple times, and what were the corresponding years, winning songs and points scored?

Old Plan:
1. Which country has won the OGAE Video Contest multiple times?
2. What were the corresponding years of these victories?

Improved Plan:
1. Which country has won the OGAE Video Contest multiple times?
2. What were the corresponding years of these victories?
3. What were the winning songs for each of these years?
4. What were the points scored in each of these victories?

Example 2:
Question: Who won the 2019 contest and what was their score?

Old Plan:
1. Who won the 2019 contest?

Improved Plan:
1. Who won the 2019 contest?
2. What was their score?

Now, please generate a plan for the following question. If an old plan is present, improve upon it as described.

Question: '''{query}'''
Old Plan: '''{old_plan}'''
"""
    messages = [
        {"role": "system", "content": prompt_system},
        {"role": "user", "content": prompt_user}
    ]
    question_list = get_completion(messages, model)
    question_list = question_list.replace("Improved Plan:\n", "")
    if '\n' in question_list:
        lines = question_list.split('\n')
    else:
        lines = re.split(r'\d+\.\s*', question_list)
    question_list = [line.strip() for line in lines if line.strip()]
    return question_list


def check_plan(query, sub_questions, model):
    prompt_system = "You are an expert in iterative plan checking."
    prompt_user = f"""Given a complex question over a table and a set of decomposed sub-questions, your task is to check whether the sub-questions are informative enough to answer the complex question.

You do NOT need to be extremely strict. As long as the sub-questions can reasonably lead to an answer that covers all the main requirements of the original question, you should return YES. Minor omissions or lack of extreme detail are acceptable, as long as the main aspects are covered.

If the plan is sufficient to answer the question, respond with YES and stop revising. Otherwise, respond with NO.

Example 1:
Question: Which country has won the OGAE Video Contest multiple times, and what were the corresponding years, winning songs and points scored?
Sub-questions:
1. Which country has won the OGAE Video Contest multiple times?
2. What were the corresponding years of these victories?
3. What were the winning songs for each of these years?
4. What were the points scored in each of these victories?

Response: YES

Example 2:
Question: Which country has won the OGAE Video Contest multiple times, and what were the corresponding years, winning songs and points scored?
Sub-questions:
1. Which country has won the OGAE Video Contest multiple times?
2. What were the corresponding years of these victories?

Response: NO

Now, please answer for the following:
Question: '''{query}'''
Sub-questions: '''{sub_questions}'''

Your response (just YES or NO):"""

    messages = [
        {"role": "system", "content": prompt_system},
        {"role": "user", "content": prompt_user}
    ]
    check = get_completion(messages, model)
    return "YES" in check.upper()

def function_generator(sub_question, table_data, model):
    prompt_system = "You are an expert Python programmer who writes functions to extract structured data from tables."
    prompt_user = f"""Your task is to write a Python function that answers a question by extracting ALL relevant information from the provided table.

**CRITICAL INSTRUCTIONS:**
1.  **Goal:** Find the relevant row(s) and extract data from ALL columns that are needed to fully answer the question.
2.  **Output Format:** The function **MUST** return a string representation of a Python dictionary (or a list of dictionaries if multiple rows are relevant). This dictionary should map column headers to their corresponding cell values.
3.  **Use `return`:** Absolutely no `print()` statements for output. The result must be returned.
4.  **No Answer Found:** If no relevant information is found, `return None`.

---
**Example 1 (Extracting specific details from one row):**
Question: What TV shows was Shagun Sharma seen in 2019?
Table Header: ["Year", "Title", "Role"]
Table Rows: [["2019", "Laal Ishq", "Pernia"], ["2019", "Vikram Betaal Ki Rahasya Gatha", "Rukmani"], ...]

**Correct Code (Returns a list of dicts):**
```python
def get_shagun_sharma_roles_2019(table):
    results = []
    for row in table['rows']:
        if row[0] == '2019':
            entry = {{
                'Title': row[1],
                'Role': row[2]
            }}
            results.append(entry)
    if results:
        return str(results)
    return None
```
*(This returns: "[{{'Title': 'Laal Ishq', 'Role': 'Pernia'}}, ...]")*

---
**Example 2 (Extracting details for a summary):**
Question: Who was the driver of the car number 23 in the 2008 Nascar Craftsman Truck Series?
Table Header: ["Rank", "No.", "Driver", "Team", "Laps", "Status"]
Table Rows: [["1", "23", "Johnny Benson", "Bill Davis Racing", "200", "Running"], ...]

**Correct Code (Returns a dict):**
```python
def get_nascar_driver_details(table):
    for row in table['rows']:
        if row[1] == '23':
            driver_details = {{
                'Driver': row[2],
                'Team': row[3]
            }}
            return str(driver_details)
    return None
```
*(This returns: "{{'Driver': 'Johnny Benson', 'Team': 'Bill Davis Racing'}}")*

---
Now, write a Python function that follows these instructions precisely.

**Question:** '''{sub_question}'''
**Table Title:** '''{table_data["title"]}'''
**Table Header:** '''{table_data["header"]}'''
**Table Rows:** '''{table_data["rows"]}'''
"""
    messages = [
        {"role": "system", "content": prompt_system},
        {"role": "user", "content": prompt_user}
    ]
    function_response = get_completion(messages, model)
    return function_response


def function_extraction(function_response, model):
    prompt_system = "You are a python function extraction assistant. The user will give you a python script."
    prompt_user = f"""Give you a paragraph about the python function, including the function body, function description, function example usage, etc. You need to extract the function body from it. Your answer only needs to contain the extracted function(start with def), no other explanation is required, no function example usage is requried.

Note that you should enclose the function with ```python\n and \n```.

Example:
'''
Idea:
1.Clarify the Question:
Firstly, we need to define the problem: from the provided data, which country has the highest number of cyclists finishing in the top 10?

2.Understand Data Structure:
Observing the structure of the given data, we noticed that each 'Cyclist' string contains a country code positioned between parentheses. Our task, then, is to extract this country code from these strings.

3.Data Extraction:
We can use string methods to extract the country code following the cyclist's name. Specifically, we utilize strip('()') to remove the parentheses.

4.Counting Cyclists per Country:
Without using Counter, we can manually tally the occurrences using a dictionary named country_counts. For each country code extracted in the previous step, if it's already in the dictionary, we increment its count; if not, we add it to the dictionary with a count of 1.

5.Determine the Most Common Country:
Using the max function and the key argument, we can identify the country with the most cyclists from the country_counts dictionary.

6.Return the Result:
The function ultimately returns the country code with the highest count of cyclists.

The following is the python script:
def top_country(table):
    countries = [row[1].strip('()') for row in table['rows']]
    country_counts = []
    for country in countries:
        if country in country_counts:
            country_counts[country] += 1
        else:
            country_counts[country] = 1
    most_common_country = max(country_counts, key=country_counts.get)
    return most_common_country
'''

Response:
```python
def top_country(table):
    countries = [row[1].strip('()') for row in table['rows']]
    country_counts = []
    for country in countries:
        if country in country_counts:
            country_counts[country] += 1
        else:
            country_counts[country] = 1
    most_common_country = max(country_counts, key=country_counts.get)
    return most_common_country
```
    
Now please give your answer. Let's follow templates of examples. 
The following is the function response:
'''
{function_response}
'''"""
    messages = [
        {"role": "system", "content": prompt_system},
        {"role": "user", "content": prompt_user}
    ]
    function_extract_response = get_completion(messages, model)
    # use regex to extract the function
    if '```python' in function_extract_response:
        function_extract_response = re.search(r'```python\n(.*?)\n```', function_extract_response, re.DOTALL).group(1)
    elif '```' in function_extract_response:
        function_extract_response = re.search(r'```\n(.*?)\n```', function_extract_response, re.DOTALL).group(1)
    else:
        function_extract_response = function_extract_response.replace('```python\n', '')
        function_extract_response = function_extract_response.replace('\n```', '')
    
    return function_extract_response

def self_debugging(sub_question, table_data, function_extract_response, feedback, model):
    prompt_system = "You are an expert in python script debugging."
    prompt_user = f"""There are some questions about the python script. You should fix the python script by given Feedback for wrong reasons. After your fixing, the python script given by you should run without error with the given table. Your answer only needs to contain the function(start with def).

Now please give your answer to debug.
The following is the question:'''{sub_question}'''
The following is the function:
'''
{function_extract_response}
'''
The following is the table:
'''
{table_data}
'''
The following is the Feedback:'''{feedback}'''"""
    messages = [
        {"role": "system", "content": prompt_system},
        {"role": "user", "content": prompt_user}
    ]
    self_debugging_response = get_completion(messages, model)
    corrected_function = function_extraction(self_debugging_response, model)
    return corrected_function


def ask_directly(question, table_data, model):
    prompt_system = "You are an expert in answering questions directly according to the given table."
    prompt_user = f"""Answer the question according to the given table. Give your short-form answer directly, no other words.

Example:
Qustion: Which team is the first?
Table:
    'header': ['Rank', 'Cyclist', 'Team', 'Time', 'UCI ProTour','Points']
    'rows': [['1','Alejandro Valverde(ESP)',"Caisse d'Epargne",'5h 29' 10"','40']

Response:
    Caisse d'Epargne

Now please give your short-form answer to the given question. Let's follow templates of examples.
The following is the question:'''{question}'''
The following is the table:'''{table_data}'''"""
    messages = [
        {"role": "system", "content": prompt_system},
        {"role": "user", "content": prompt_user}
    ]
    response = get_completion(messages, model)
    return response

def sentence_generator(short_answer, sub_question, model):
    prompt_system = "You are an expert in generating natural language sentences from structured data, adapting your style to the context."
    prompt_user = f"""Your task is to convert the structured 'Data' into a single, high-quality, factual sentence that answers the 'Question'.
The style of the sentence should mimic the provided examples.

**INSTRUCTIONS:**
1.  Carefully analyze the 'Data' (which is a dictionary or a list of dictionaries).
2.  Synthesize all information from the 'Data' into a flowing, natural sentence.
3.  Ensure the sentence directly and completely answers the 'Question'.
4.  Do not add information not present in the 'Data'.

---
**Example 1 (Concise & Detail-Oriented):**
Question: What TV shows was Shagun Sharma seen in 2019?
Data: `[{{'Title': 'Laal Ishq', 'Role': 'Pernia'}}, {{'Title': 'Vikram Betaal Ki Rahasya Gatha', 'Role': 'Rukmani'}}, {{'Title': 'Shaadi Ke Siyape', 'Role': 'Dua'}}]`

**Generated Sentence:**
In 2019, Shagun Sharma appeared in Laal Ishq as Pernia, Vikram Betaal Ki Rahasya Gatha as Rukmani, and Shaadi Ke Siyape as Dua.

---
**Example 2 (More Descriptive Summary):**
Question: Who was the driver of the car number 23 in the 2008 Nascar Craftsman Truck Series?
Data: `{{'Driver': 'Johnny Benson', 'Team': 'Bill Davis Racing'}}`

**Generated Sentence:**
The driver of car number 23 in the 2008 Nascar Craftsman Truck Series was Johnny Benson, who drove for Bill Davis Racing.

---
Now, generate the sentence based on the following inputs.

**Question:** '''{sub_question}'''
**Data:** '''{short_answer}'''
"""
    messages = [
        {"role": "system", "content": prompt_system},
        {"role": "user", "content": prompt_user}
    ]
    long_answer = get_completion(messages, model)
    return long_answer

def generate_final_answer_fetaqa(query, answer_list, model):
    prompt_system = "You are an expert answer generator that produces concise, direct answers."
    prompt_user = f"""You will be given a main 'Question' and a 'Fact List'. Your task is to synthesize these facts into a final, concise answer.

**STYLE GUIDE:**
Generate a direct, concise answer in one sentence. Be factual and to the point, avoiding unnecessary elaboration.

**Examples:**
Question: Who won the 1982 Illinois gubernatorial election, and how many votes was the margin?
Answer: Thompson prevailed in the 1982 Illinois gubernatorial election by a 5,074 vote margin.

Question: How did Michael and Mario Andretti do?
Answer: Michael Andretti finished with a run of 214.522 mph, faster than Mario.

Question: How many copies did "Pleasure" sell in 1998 alone, and how long was it the best selling album in Japan?
Answer: B'z The Best "Pleasure" sold more than 5 million copies in 1998 alone, making it a temporary best-selling album in Japanese music history, until being surpassed by Utada Hikaru's First Love in 1999.

Question: How many passengers can that plane hold?
Answer: The Tigerair Australia fleet consists of the following aircraft and Virgin Australia announced that the entire A320 fleet will be replaced with Boeing 737-800 aircraft.

Question: When and in what play did Platt appear at the Music Box Theatre?
Answer: In 2016 and 2017, Platt played in Dear Evan Hansen on Broadway at the Music Box Theatre.

**Universal Rule:** NEVER invent information. Your answer MUST be grounded exclusively in the provided 'Fact List'.

**Question:** '''{query}'''
**Fact List:** '''{answer_list}'''
"""
    messages = [
        {"role": "system", "content": prompt_system},
        {"role": "user", "content": prompt_user}
    ]
    final_answer = get_completion(messages, model)
    return final_answer

def generate_final_answer_qtsumm(query, answer_list, model):
    prompt_system = "You are an expert summary generator that produces comprehensive, flowing summaries."
    prompt_user = f"""You will be given a main 'Query' and a 'Fact List'. Your task is to synthesize these facts into a final, comprehensive summary.

**STYLE GUIDE:**
Generate a well-structured, flowing paragraph (50-60 words) that comprehensively addresses the query. Use connecting words and phrases to create coherent, readable text.

**Examples:**
Query: Summarize the basic information of the episodes of Dragon Zakura aired in July.
Summary: The Dragon Zakura TV series aired multiple episodes in July 2005. The first episode of July titled "Know Your Weaknesses!" had a rating of 16.5%, while the following two episodes had ratings of 13.8% and 16.1%. The second episode of the month was titled "Entrance Exams Are Sports , So Play!". The last episode was named "Hold Out Until You Hit The Wall".

Query: Which drivers and their corresponding teams achieved the top five best speeds during the 2018 Chevrolet Silverado 250 qualifying results and what was the manufacturer of their vehicles?
Summary: In 2018 Chevrolet Silverado 250 qualifying results, top five best speed by these drivers and their teams: Ben Rhodes with ThorSport Racing get highest speed 97.854; after him is Noah Gragson with Kyle Busch Motorsports at 97.395. Myatt Snider from ThorSport Racing get 96.610, Matt Crafton also from ThorSport Racing get 95.971, and last, Timothy Peters with GMS Racing get 95.696 as fifth best speed. These top drivers' cars made by Ford for Rhodes, Snider, and Crafton; Gragson drive Toyota, Peters drive Chevrolet.

Query: Did any athlete from the United States participate in the Women's 50 metre freestyle final at the 2012 Summer Olympics, and if so, what was her rank and final time?
Summary: Yes, an athlete from the United States participated in the Women's 50 metre freestyle final at the 2012 Summer Olympics-Jessica Hardy. She finished with a rank of 7th place with a final time of 24.62 seconds.

**Universal Rule:** NEVER invent information. Your summary MUST be grounded exclusively in the provided 'Fact List'.

**Query:** '''{query}'''
**Fact List:** '''{answer_list}'''
"""
    messages = [
        {"role": "system", "content": prompt_system},
        {"role": "user", "content": prompt_user}
    ]
    final_answer = get_completion(messages, model)
    return final_answer
