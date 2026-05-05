# !nvidia-smi

#!pip install crewai

#!pip install 'crewai[tools]'
# !pip install crewai-tools
# !pip install crewai-tools
# !pip install --upgrade crewai crewai-tools
# !pip install crewai langchain-google-genai
# pip install litellm
# pip uninstall crewai litellm -y
# pip install crewai litellm


from re import VERBOSE
import os
from crewai import Agent, Task, Crew
from crewai_tools import SerperDevTool
from langchain_google_genai import ChatGoogleGenerativeAI

os.environ["LITELLM_PROVIDER"] = "gemini"
# os.environ['OPENAI_API_KEY'] = 'use your api key'
os.environ['SERPER_API_KEY'] = ''
os.environ["GOOGLE_API_KEY"] = ''
search_tool = SerperDevTool()

llm = "gemini/gemini-2.5-flash"
teacher = Agent(
    role='Teacher',
    goal='Explain concepts clearly step by step',
    backstory='Experienced teacher who explains with examples',
    llm=llm,
    verbose=True
)

researcher = Agent(role = 'Researcher',
                   goal='Find latest information',
                   backstory = 'Expert in searching real-world data',
                   tools=[search_tool],
                   llm = llm,
                   verbose = True)

simplifier = Agent(role = 'Simplifier',
                   goal = 'Make thing easy',
                   backstory = 'Breaks complex ideas into simple language',
                   llm = llm,
                   verbose=True)

student = Agent(role= 'Student',
                goal = 'Take notes',
                backstory = 'Writes simple notes like learner',
                llm = llm,
                verbose = True)

examiner = Agent (role = 'Examiner',
                 goal= 'Create questions',
                 backstory = 'Tests understanding',
                 llm = llm,
                 verbose=True)

topic = 'What is difference between agents vs agentic ai vs mcp vs generative ai'

task1 = Task(
    description = f'Explain{topic} in simple steps with examples',
    expected_output= 'step-by-step explanation',
    agent = teacher

)
task2 = Task(
    description = f'search and find 3 important points about {topic}',
    expected_output = '3 clear points',
    agent = researcher
)

task3 = Task(
    description = f'Simplify the explanation of {topic}',
    expected_output='Very simple explanation',
    agent = simplifier
)

task4 = Task(
    description=f'Write short notes on {topic} like a student',
    expected_output='Simple notes',
    agent= student

)

task5 = Task(
    description=f'Create 5 questions about {topic}',
    expected_output='3 easy questions',
    agent=examiner

)


crew = Crew(
    agents=[teacher, researcher, simplifier, student, examiner],
    tasks=[task1, task2, task3, task4, task5],
    verbose=True
)

result = crew.kickoff()
print(result)
