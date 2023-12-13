from fastapi import FastAPI
from transformers import pipeline
from pydantic import BaseModel
import uvicorn

class QADataModel(BaseModel):
     question: str
     context: str

model_name = 'distilbert-base-cased-distilled-squad'
model = pipeline(model=model_name, tokenizer=model_name,
                          task='question-answering')

app = FastAPI()

@app.post("/question_answering")

async def qa(input_data: QADataModel):
     result = model(question = input_data.question,
                    context=input_data.context)
     return {"result": result["answer"]}