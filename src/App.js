import React, { useState } from 'react';
import ImageGenerator from './components/ImageGenerator';
import ImageDisplay from './components/ImageDisplay';

function App() {
  const [generatedImage, setGeneratedImage] = useState(null);

  const handleGenerate = async (query) => {
    try {
      const response = await fetch('http://localhost:5000/generate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query }),
      });
      const data = await response.json();
      setGeneratedImage(data.image_url);
    } catch (error) {
      console.error('Error generating image:', error);
    }
  };

  return (
    <div className="App">
      <h1>Image Generator</h1>
      <ImageGenerator onGenerate={handleGenerate} />
      <ImageDisplay imageUrl={generatedImage} />
    </div>
  );
}

export default App;