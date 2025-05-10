from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from app.agent import load_PDF_doc, run_compliance_query, setup_compliance_qa_chain
import tempfile, shutil, os, uvicorn

app = FastAPI()


qa_chain, vectorstore = setup_compliance_qa_chain()

@app.post("/check")
async def check(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        query = load_PDF_doc(tmp_path)[:8000]  
        response = run_compliance_query(query, qa_chain)
    finally:
        os.remove(tmp_path)

    return JSONResponse(content=response)


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
