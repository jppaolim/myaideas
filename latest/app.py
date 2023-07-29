from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from myaideas import main
from pydantic import BaseModel

class Query(BaseModel):
    query: str

app = FastAPI()

templates = Jinja2Templates(directory="templates")

@app.post("/")
def index(query: Query, request: Request):
    results = main(query.query)
    return templates.TemplateResponse('index.html', {"request": request, "results": results})

@app.get("/")
def read_root(request: Request):
    return templates.TemplateResponse('index.html', {"request": request})
