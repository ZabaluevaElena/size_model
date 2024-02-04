#!/usr/bin/env python3

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

import joblib
import pandas as pd
import uvicorn

app = FastAPI()
model = joblib.load("model/model.joblib")
SIZE_MAP: dict = {1: "XXS", 2: "S", 3: "M", 4: "L", 5: "XL", 6: "XXXL"}
templates = Jinja2Templates("templates")


@app.get("/")
def root(request: Request):
    return templates.TemplateResponse(name="index.html", context={"request": request})


@app.post("/postdata")
async def predict(
    request: Request, weight: int = Form(), age: int = Form(), height: int = Form()
):
    X = pd.DataFrame([{"weight": weight, "age": age, "height": height}])
    try:
        n_pred = model.predict(X)[0]
        p_pred = model.predict_proba(X).max()
    except Exception as e:
        return HTMLResponse(content=str(e), status_code=500)
    pred_size = SIZE_MAP.get(n_pred)
    return templates.TemplateResponse(
        name="response.html",
        context={"request": request, "predict": pred_size, "p_pred": p_pred},
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
