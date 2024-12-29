import gradio as gr
from dotenv import load_dotenv
from Answer_Service import AiGenerateAnswerService
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from Tools import get_querying_postgres_tool, get_custom_movies_api_tool
from sqlalchemy.exc import SAWarning
import warnings
import os
from langchain_core.messages import AIMessage
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from sqlalchemy import create_engine
from langchain.agents import create_openai_tools_agent
from langchain.agents.agent import AgentExecutor
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
import random
import string

load_dotenv()

OPENAI_API_KEY =  os.environ.get("OPENAI_API_KEY")
LLM = ChatOpenAI(model="gpt-4o", temperature=0)
warnings.filterwarnings("ignore", category=SAWarning)
memory = ChatMessageHistory(session_id="test-session")
tools = [get_querying_postgres_tool(), get_custom_movies_api_tool()]

def QueryingPostgresTool(query: str):
    db_config = {
        "dbname": os.environ.get("DB_NAME_V2"),
        "user": os.environ.get("DB_USER_V2"),
        "password": os.environ.get("DB_PASSWORD_V2"),
        "host": os.environ.get("DB_HOST_V2", "localhost"),
        "port": os.environ.get("DB_PORT", "5432")
    }
    connection_string = (
        f"postgresql+psycopg2://{db_config['user']}:{db_config['password']}"
        f"@{db_config['host']}:{db_config['port']}/{db_config['dbname']}"
    )
    engine = create_engine(connection_string)
    try:
        db = SQLDatabase(engine)
    except Exception as e:
        error_message = f"Error connecting to the database: {e}"
        print(error_message)
        raise SystemExit(error_message)
    toolkit = SQLDatabaseToolkit(db=db, llm=LLM)
    context = toolkit.get_context()
    tools = toolkit.get_tools()
    SQL_FUNCTIONS_SUFFIX = """
    You are provided with the following tools for interacting with a SQL database. Each tool serves a specific purpose, and it's important to know when and how to use them correctly. Below is a description of each tool, along with guidelines on when to use each tool.

    1. **ListSQLDatabaseTool (sql_db_list_tables)**:
       - **Description**: This tool retrieves a comma-separated list of all table names from the database. It does not take any input and simply returns the full list of tables.
       - **When to use**: Use this tool at the beginning of a query session to list all available tables in the database. This helps you understand what data is available in the database and helps identify which tables might be relevant to the user's query.
    
    2. **InfoSQLDatabaseTool (sql_db_schema)**:
       - **Description**: This tool retrieves the schema (i.e., column names and data types) for specific tables in the database. You must provide a comma-separated list of table names to this tool. It returns the schema for the specified tables along with some sample rows of data.
       - **When to use**: After identifying which tables might be relevant to a user's query, use this tool to get detailed schema information for those specific tables. It is **crucial** that you carefully identify all tables that could possibly be relevant to the user's query. Be **very accurate** in this step, ensuring that no potentially relevant table is overlooked. Include every possible relevant table, even if there is only a slight chance that it may contain useful data for the query. Missing a relevant table at this step could result in incomplete or inaccurate query results.
    
    3. **QuerySQLDataBaseTool (sql_db_query)**:
       - **Description**: This tool executes a SQL query against the database and returns the result. You must provide a syntactically correct SQL query as input. If the query is incorrect or fails, it will return an error message instead of throwing an exception.
       - **When to use**: Use this tool when you are ready to execute a SQL query and retrieve actual data from the database. Always make sure the query is correct before using this tool to avoid errors. If an error occurs, you should revise the query and try again.
    
    4. **QuerySQLCheckerTool (sql_db_query_checker)**:
       - **Description**: This tool uses a language model (LLM) to check the correctness of a SQL query before executing it. You provide the SQL query as input, and the tool will analyze it to ensure it is valid for execution. It checks for syntax errors, dialect issues, and correctness.
       - **When to use**: Before executing any SQL query using the `sql_db_query` tool, use this tool to validate the query. This is especially useful when you're unsure if the query is correct or if the query might result in an error. It acts as a safety check before running the actual query.
    
    ---
    
    ### Guidelines for Use:
    - **Start with `sql_db_list_tables`**: Always begin by listing all available tables in the database. This will give you an overview of the data sources you can query.
    - **Be very accurate with `sql_db_schema`**: Once you have identified which tables might be relevant to the user’s query, use `sql_db_schema` to fetch the schema of those tables. In this step, it is essential to be **very precise** when deciding which tables are relevant. Include **all possible relevant tables** and do not ignore any that might contain useful data. Ensure that the query will have access to all relevant information, even from tables that might seem only marginally related to the query.
    - **When generating SQL queries, always incorporate flexible search patterns (e.g., LIKE, ILIKE, SOUNDEX, or wildcard %), especially when users ask for data but are unsure of the exact formatting, casing, or content of the data. This ensures that searches are not limited by precise matches, and instead capture a wider range of potential results. For text-based columns, use ILIKE or LIKE with the % wildcard for partial matching. If dealing with phonetic similarity, consider using SOUNDEX for better matches based on sound.**
    - **Check the query with `sql_db_query_checker`**: Before executing the final SQL query, run it through the `sql_db_query_checker` to ensure it is correct and valid.
    - **Execute the query with `sql_db_query`**: After confirming the query is correct, use `sql_db_query` to retrieve the data and provide the final result to the user.
    
    Always ensure that the SQL query is crafted based on the schema information, and double-check for errors using the query checker tool before execution.

"""
    messages = [
        HumanMessagePromptTemplate.from_template("{input}"),
        AIMessage(content=SQL_FUNCTIONS_SUFFIX),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
    prompt = ChatPromptTemplate.from_messages(messages)
    prompt = prompt.partial(**context)
    agent = create_openai_tools_agent(llm=LLM, tools=tools, prompt=prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=toolkit.get_tools(),
        verbose=True,
        max_iterations=20,
    )
    response = agent_executor.invoke({"input": query})
    return [AIMessage(content=f"The result is: {response}")]

def get_agent_answer(params: dict):
    question_id = params.get('question_id')
    query = params.get('input')
    history = params.get('history')
    flag = params.get('flag')
    session_id = params.get('session_id')
    chatbott = AiGenerateAnswerService("30")   # To change
    Document_answers = chatbott.generate_answer(question_id, query, history, flag)
    db_answer = QueryingPostgresTool(query)
    if db_answer is None:
        db_answer = []
    params['Document_answers'] = Document_answers
    params['db_answer'] = db_answer
    system_template = """
    You are an AI with access to historical context. You are designed to provide a final answer based on the combination of two sources:
    (1) the answer derived from the user's documents (PDF answers) stored in {Document_answers}, and
    (2) the answer derived from querying the database through {db_answer}, which comes from the QueryingPostgresTool.
    Your task is to generate a final answer that combines insights from both sources. You should always embed the information from both answers into your response. If both answers provide relevant information, merge them coherently. If only one answer provides relevant information, focus on that source.
    Always prioritize combining both answers to generate a comprehensive response that addresses the user's question accurately. However, if only one answer is sufficient to fully address the user's question, rely on it while acknowledging the other source when relevant.
    DO NOT rely on your own knowledge for answering the question, and only use the data provided from {Document_answers} and {db_answer}. Additionally, use the {chat_history} to maintain the context of the conversation.
    Here is the chat history so far:
    {chat_history}
    Here is the user's query:
    {input}
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_template),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="Document_answers"),
            MessagesPlaceholder(variable_name="db_answer"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
            MessagesPlaceholder(variable_name="chat_history"),
        ]
    )

    agent = create_openai_tools_agent(llm=LLM, tools=tools, prompt=prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
    agent_with_chat_history = RunnableWithMessageHistory(
        agent_executor,
        lambda session_id: memory,
        input_messages_key="input",
        history_messages_key="chat_history",
    )
    params['chat_history'] = memory.messages
    params['Document_answers'] = Document_answers
    agent_answer = agent_with_chat_history.invoke(params,
                                                  config={"configurable": {"session_id": session_id}},
                                                  )

    print("Agent Answer:", agent_answer)
    return agent_answer

used_question_ids = set()

def generate_session_name():
    return ''.join(random.choices(string.ascii_letters + string.digits, k=10))

def generate_question_id(session_question_ids):
    while True:
        try:
            question_id = random.randint(10 ** 10, 10 ** 14)  # Ensure uniqueness
            if question_id not in used_question_ids and question_id not in session_question_ids:
                used_question_ids.add(question_id)
                session_question_ids.add(question_id)
                return question_id
        except Exception as e:
            return f"An unexpected error occurred: {str(e)}"


def handle_user_query(input_text, session_name, session_question_ids):
    if not session_name:
        session_name = generate_session_name()
    question_id = str(generate_question_id(session_question_ids))

    params = {
        'question_id': question_id,
        'input': input_text,
        'history': None,
        'flag': False,
        'session_id': session_name,
    }

    try:
        agent_answer = get_agent_answer(params)
        if isinstance(agent_answer, dict) and 'type' in agent_answer and agent_answer['type'] == 'error':
            return f"Error: {agent_answer['data']}", session_name, session_question_ids
        return agent_answer["output"], session_name, session_question_ids
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}", session_name, session_question_ids


def clear_session():
    return "", generate_session_name(), set()

with gr.Blocks() as demo:
    session_state = gr.State(value=generate_session_name())
    session_question_ids_state = gr.State(value=set())

    gr.Markdown("# Release Two")

    user_input = gr.Textbox(label="Enter your question")
    output_text = gr.Textbox(label="Agent Answer", interactive=False)

    with gr.Row():
        send_btn = gr.Button("Get Answer")
        clear_btn = gr.Button("Clear")

    send_btn.click(
        handle_user_query,
        inputs=[user_input, session_state, session_question_ids_state],
        outputs=[output_text, session_state, session_question_ids_state]
    )

    clear_btn.click(
        clear_session,
        inputs=[],
        outputs=[output_text, session_state, session_question_ids_state]
    )

demo.launch()

