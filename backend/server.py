from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from load_model import *
from fastapi.responses import StreamingResponse

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3001"],  # Allow requests from the frontend
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (POST, GET, OPTIONS, etc.)
    allow_headers=["*"],  # Allow all headers
)


class PromptRequest(BaseModel):
    description: str
    #color: str = None  # Optional extra fields
    #fabric: str = None

@app.post("/generate")
async def generate_image(request: PromptRequest , num_samples: int = 5):
    # Print the received data
    print(f"Description: {request.description}")
    caption = request.description
    device = 'cpu'
    input_ids = tokenizer(caption,padding='max_length', truncation=True, max_length=77,return_tensors='pt')['input_ids'].to(device)
    
    # Generate images using the model
    generated_images = test_model(galip_container, input_ids, device, num_samples=num_samples)
    
    # Send back the first image as an example (you can send multiple as needed)
    return StreamingResponse(generated_images[0], media_type="image/png")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
