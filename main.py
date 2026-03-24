
from fastapi import FastAPI
from chainlit.utils import mount_chainlit
import uvicorn


app = FastAPI()




mount_chainlit(app=app, target="app.py", path="/")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)