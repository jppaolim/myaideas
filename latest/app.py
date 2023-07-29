from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from myaideas import main

app = FastAPI()

templates = Jinja2Templates(directory="templates")

@app.post("/")
def index(request: Request, query: str):
    results = main(query)
    return templates.TemplateResponse('index.html', {"request": request, "results": results})

@app.get("/")
def read_root(request: Request):
    return templates.TemplateResponse('index.html', {"request": request})
