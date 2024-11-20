import os
from model import * 
import torchvision.transforms as transforms
from transformers import CLIPTokenizer
from io import BytesIO

CLIP, _ = CLIP_Model.get_model()  # Assuming CLIP model is defined elsewhere
model_singleton = GalipModelSingleton('state_epoch_080_000_epochs_optimized.pth', CLIP)
galip_container, tokenizer = model_singleton.get_model()

def preprocess(caption:str) :
    
    tokens = tokenizer(caption, padding='max_length', truncation=True, max_length=77, return_tensors="pt")
    input_ids = tokens.input_ids.to("cpu")
    return input_ids


def test_model(galip_obj, input_ids, device, z_dim=100, num_samples=5):
    galip_obj.netG.eval()  # Set the generator to evaluation mode
    transform = transforms.ToPILImage()  # Convert the image tensor to PIL format
    images = []  # List to store generated images
    os.makedirs("gen_images", exist_ok=True)
    with torch.no_grad():  # No need to calculate gradients during testing
        for i in range(num_samples):
            # Generate random noise
            noise = torch.randn(1, z_dim).to(device)
            
            # Get caption embeddings using the text encoder
            caption_emb, word_embeds = galip_obj.text_encoder(input_ids)
            
            # Generate a fake image using the generator
            fake_image = galip_obj.netG(noise, caption_emb)
            # Assuming the generated image is normalized [-1, 1], denormalize it to [0, 1]
            fake_image = (fake_image.squeeze(0) * 0.5 + 0.5).clamp(0, 1)
            image_path = os.path.join("gen_images", f"generated_image_{i+1}.png")
            # Convert to PIL image and store it in memory
            pil_image = transform(fake_image)
            pil_image.save(image_path)  # Save the image locally
            img_byte_arr = BytesIO()
            pil_image.save(img_byte_arr, format='PNG')
            
            img_byte_arr.seek(0)  # Move to the start of the stream
            images.append(img_byte_arr)
            images.append(pil_image)
    
    # Return the list of generated images
    return images
