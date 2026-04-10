from dotenv import load_dotenv
from transformers import pipeline
from langchain.llms import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain.schema.runnable import RunnableBranch, RunnableLambda
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()
pipe = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    max_length=100
)

model = HuggingFacePipeline(pipeline=pipe)
parser = StrOutputParser()

class Feedback(BaseModel):
    sentiment: Literal['positive', 'negative'] = Field(
        description='Give the sentiment of the feedback'
    )

parser2 = PydanticOutputParser(pydantic_object=Feedback)
prompt1 = PromptTemplate(
    template=(
        "Classify sentiment as positive or negative.\n"
        "Text: {feedback}\n"
        "{format_instruction}"
    ),
    input_variables=['feedback'],
    partial_variables={
        'format_instruction': parser2.get_format_instructions()
    }
)

classifier_chain = prompt1 | model | parser2
prompt2 = PromptTemplate(
    template="Write a short reply to this positive feedback:\n{feedback}",
    input_variables=['feedback']
)

prompt3 = PromptTemplate(
    template="Write a short reply to this negative feedback:\n{feedback}",
    input_variables=['feedback']
)

branch_chain = RunnableBranch(
    (lambda x: x.sentiment == 'positive', prompt2 | model | parser),
    (lambda x: x.sentiment == 'negative', prompt3 | model | parser),
    RunnableLambda(lambda x: "Could not determine sentiment")
)

chain = classifier_chain | branch_chain

result = chain.invoke({
    'feedback': 'This is a beautiful phone'
})

print("\nFinal Output:\n", result)

print("\n\nGraph:\n")
chain.get_graph().print_ascii()