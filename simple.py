print("Hello World!")

from fastapi import FastAPI
import gpt_2_simple as gpt2  #for gpt2 prediction 

app = FastAPI()

# put below in terminal to launch uvicorn server
# uvicorn simple:app --reload

# define a root `/` endpoint
@app.get("/")
def index():
    return {"ok": True}

# adding a predict endpoint to our API
@app.get("/generate")
def generate(prompt="Once upon a time"):
    # generates text based on our gpt-2 model and prompt
    # need to change the length between calls or it returns an error
    sess = gpt2.start_tf_sess()
    gpt2.load_gpt2(sess)
    gen_text = gpt2.generate(sess,
            prefix = prompt,
            length = 20,
            nsamples = 3,
            return_as_list=True
            )
    return {'text' : gen_text}
    
