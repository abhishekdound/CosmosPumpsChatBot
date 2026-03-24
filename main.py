
from fastapi import FastAPI
from chainlit.utils import mount_chainlit
import uvicorn


app = FastAPI()




mount_chainlit(app=app, target="app.py", path="/")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
