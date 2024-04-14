from pydantic import BaseModel

from DocumentController import DocumentController
from generation.openai_chat import ChatOpenAI
from util import setup_logging

from fastapi import FastAPI, UploadFile, HTTPException, status
import uvicorn
import logging
import os
from dotenv import load_dotenv

load_dotenv()
setup_logging()
logger = logging.getLogger(__name__)
app = FastAPI()

controller = DocumentController()
generate = ChatOpenAI()


logger.info("Starting...")
# file_path = 'data/pdf/amazon-dynamo-sosp2007.pdf'
# pkl_file_name = ''.join(file_path.split('/')[-1].split('.')[:-1]) + '.pkl'
# pkl_file_path = os.path.join('data', 'pkl', pkl_file_name)


@app.post("/upload-pdf/")
async def upload_pdf_file(file: UploadFile):
    if file.content_type != 'application/pdf':
        raise HTTPException(status_code=400, detail="Invalid file type. Only PDF files are accepted.")

    try:
        # Save the uploaded file to a directory
        logger.debug(f"Got file {file.filename}. Saving..")
        file_location = f"data/pdf/{file.filename}"
        os.makedirs(os.path.dirname(file_location), exist_ok=True)
        with open(file_location, "wb+") as file_object:
            file_object.write(await file.read())

        await controller.process_text_and_store(file_location)
        return status.HTTP_201_CREATED
    except Exception as e:
        return {"error": str(e)}


@app.delete("/indexes")
async def delete_index():
    controller.delete_indexes()
    return status.HTTP_202_ACCEPTED


class Question(BaseModel):
    question: str


@app.post("/question")
async def question_answer(body: Question):
    context = controller.search_and_retrieve_result(body.question)

    answer = generate.get_message(body.question, context, model="gpt-4-turbo")
    # answer = generate.get_message(body.question, context)

    return {"answer": answer}


# query = "What is N, R and W?"


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
