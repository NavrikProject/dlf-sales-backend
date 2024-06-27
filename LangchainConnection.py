import os
import urllib
import pyodbc
from dotenv import load_dotenv
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.orm import sessionmaker
from langchain_community.utilities import SQLDatabase
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain.chains import create_sql_query_chain
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain.memory import ConversationBufferMemory, ChatMessageHistory
from langchain_core.runnables import RunnablePassthrough, RunnableSequence
from operator import itemgetter
import pandas as pd
from sqlalchemy import create_engine, MetaData, Table
import logging

load_dotenv()

sql_database_user = os.getenv("sql_database_user")
sql_database_pass = os.getenv("sql_database_pass")
sql_database_server = os.getenv("sql_database_server")
sql_database_name = os.getenv("sql_database_name")


openai_key = os.getenv("OPENAI_API_KEY")
langchain_tracing = os.getenv("LANGCHAIN_TRACING_V2")
langchain_key = os.getenv("LANGCHAIN_API_KEY")

def get_chain():
    # URL encode the password
    #db_pass_encoded = urllib.parse.quote_plus(sql_database_pass)
    # Create the ODBC connection string
    odbc_str = (
        f'DRIVER={{ODBC Driver 18 for SQL Server}};'
        f'SERVER={sql_database_server};'
        f'DATABASE={sql_database_name};'
        f'UID={sql_database_user};'
        f'PWD={sql_database_pass}'
    )
    # Create the SQLAlchemy connection string
    connection_string = f'mssql+pyodbc:///?odbc_connect={urllib.parse.quote_plus(odbc_str)}'
    # Create SQLAlchemy engine
    engine = create_engine(connection_string)
    print(engine)
    # Create SQLAlchemy MetaData instance
    metadata = MetaData()
    # Reflect only the 'transaction' table
    transaction_table = Table('transaction', metadata, autoload_with=engine, schema="dbo")
    # Create a LangChain SQLDatabase instance for only the 'transaction' table
    db = SQLDatabase(engine, include_tables=['transaction'])
    LLM = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.5)
    generate_query = create_sql_query_chain(LLM, db)
    execute_query = QuerySQLDataBaseTool(db=db)
    answer_prompt = PromptTemplate.from_template(
        """Given the following user question, corresponding SQL query, and SQL result, answer the user question.
        Question: {question}
        SQL Query: {query}
        SQL Result: {result}
        Answer: """
    )

    # Set up the rephrase answer pipeline
    rephrase_answer = answer_prompt | LLM | StrOutputParser()

    chain = (
        RunnablePassthrough.assign(query=generate_query).assign(
            result=itemgetter("query") | execute_query
        ) | rephrase_answer
    )
    return chain

def create_history(messages):
    history = ChatMessageHistory()
    for message in messages:
        if message["role"] == "user":
            history.add_user_message(message["content"])
        else:
            history.add_ai_message(message["content"])
    return history

def invoke_chain(question, messages):
    chain = get_chain()
    history = create_history(messages)
    response = chain.invoke({"question": question, "messages": history.messages})
    history.add_user_message(question)
    history.add_ai_message(response)
    return response
