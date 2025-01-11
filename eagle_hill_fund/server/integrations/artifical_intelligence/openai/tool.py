import os
from dotenv import (
    load_dotenv,
)

from eagle_hill_fund.server.integrations.artifical_intelligence.tool import LLMBaseClass

load_dotenv()

from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents.agent_types import AgentType
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import TextSplitter
from openai import OpenAI


class OpenAITool(LLMBaseClass):
    def __init__(self, pandas_agent: bool = False):
        super().__init__()
        self.open_ai_api_key = os.getenv("OPEN_AI_API_KEY")

        if pandas_agent:
            self.pandas_agent = None

        self.client = OpenAI(api_key=self.open_ai_api_key)

    def initialize_pandas_agent(self, df):
        """

        :param df:
        :return:
        """
        self.pandas_agent = create_pandas_dataframe_agent(
            ChatOpenAI(temperature=0.7, model="gpt-3.5-turbo-0613", openai_api_key=self.open_ai_api_key),
            df,
            verbose=True,
            agent_type=AgentType.OPENAI_FUNCTIONS,
            return_intermediate_steps=False,
        )

    def ask_question_about_df(self, question: str = None):
        """
        Ask a conversational question about the DataFrame and maintain conversation context.

        :param question:
        :return:
        """
        # Format the question for conversational insight
        conversational_question = f"{question}"

        # Run the agent to answer the question about the DataFrame
        response = self.pandas_agent.invoke(input={"input": conversational_question}, handle_parsing_errors=True)[
            "output"
        ]

        return response

    def ask_question(self, question):
        raw_response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": question,
                },
            ],
            temperature=0.25,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
        )
        return raw_response.choices[0].message.content

    def summarize_text(self, text):
        """
        Sends text to the OpenAI API to generate a summary.
        """
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that summarizes text."},
                {"role": "user", "content": f"Summarize the following text:\n\n{text}"},
            ],
            max_tokens=500,
            temperature=0.25,  # Keeps the summary concise and factual
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
        )
        return response.choices[0].message.content  # Corrected access to the content

    def summarize_large_article(self, article_text):
        """
        Splits and summarizes a long article, then combines the summaries.
        """
        chunks = self.split_text(article_text, max_tokens=3000)
        summaries = []

        for i, chunk in enumerate(chunks):
            summary = self.summarize_text(chunk)
            summaries.append(summary)

        # Combine chunk summaries into a final summary
        final_summary = self.summarize_text(" ".join(summaries))
        return final_summary
