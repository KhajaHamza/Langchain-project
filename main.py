from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

def generate_company_name(product):
    llm = OpenAI(temperature=0.7)
    prompt = PromptTemplate(
        input_variables=["product"],
        template="What is a good name for a company that makes {product}?",
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run(product)

if __name__ == "__main__":
    product = "eco-friendly water bottles"
    company_name = generate_company_name(product)
    print(f"Suggested company name: {company_name}")
