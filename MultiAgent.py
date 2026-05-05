
import os
import gradio as gr
from crewai import Agent, Task, Crew
from crewai_tools import SerperDevTool

os.environ['SERPER_API_KEY'] = ''
os.environ["GOOGLE_API_KEY"] = ''
search_tool = SerperDevTool()

def run_multi_agent(topic):
  teacher = Agent(role ='Teacher',
                goal ='Explain concepts clearly step by step',
                backstory = 'Experienced teacher who explain with example',
                llm = 'gpt-4o-mini',
                verbose = True)
  researcher = Agent(role = 'Researcher',
                   goal='Find latest information',
                   backstory = 'Expert in searching real-world data',
                   tools=[search_tool],
                   llm = 'gpt-4o-mini',
                   verbose = True)
  simplifier = Agent(role = 'Simplifier',
                   goal = 'Make thing easy',
                   backstory = 'Breaks complex ideas into simple language',
                   llm = 'gpt-4o-mini',
                   verbose=True)
  student = Agent(role= 'Student',
                goal = 'Take notes',
                backstory = 'Writes simple notes like learner',
                llm = 'gpt-4o-mini',
                verbose = True)
  examiner = Agent (role = 'Examiner',
                 goal= 'Create questions',
                 backstory = 'Tests understanding',
                 llm = 'gpt-4o-mini',
                 verbose=True)

  task1 = Task(
    description = f'Explain{topic} in simple steps with examples',
    expected_output= 'step-by-step explanation',
    agent = teacher)

  task2 = Task(
    description = f'search and find 3 important points about {topic}',
    expected_output = '3 clear points',
    agent = researcher)

  task3 = Task(
    description = f'Simplify the explanation of {topic}',
    expected_output='Very simple explanation',
    agent = simplifier )

  task4 = Task(
    description=f'Write short notes on {topic} like a student',
    expected_output='Simple notes',
    agent= student)

  task5 = Task(
    description=f'Create 5 questions about {topic}',
    expected_output='3 easy questions',
    agent=examiner)

  crew = Crew(
      agents=[teacher, researcher, simplifier, student, examiner],
      tasks=[task1, task2, task3, task4, task5],
      verbose=False)

  result = crew.kickoff()

  return str(result)

interface = gr.Interface(
    fn = run_multi_agent,
    inputs = gr.Textbox(
        label = 'Enter Topic',
        placeholder = 'what is Agentic AI vs single '),
    outputs = gr.Textbox(label = 'Multi-Agent Output'),
    title = 'Jagatjyoti Mohanty MultiAgent AI Training',
    description = 'Enter any topic and see multiple AI agents colabotate to teach, simplify and test you'

)

if __name__ == '__main__':
  interface.launch()
