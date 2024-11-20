import requests

# Send POST request with data to your API endpoint
response = requests.post("http://127.0.0.1:8000/generate", json={"description": "Golden Colour Silk Wedding Sherwani With Zari,Thread,Hand Work. Golden Silk"})

# Save the image if the response is successful
if response.status_code == 200:
    with open("generated_image.png", "wb") as f:
        f.write(response.content)
    print("Image saved as 'generated_image.png'")
else:
    print(f"Failed to retrieve image, status code: {response.status_code}")
