import React from 'react';

function ImageDisplay({ imageUrl }) {
  return (
    <div>
      {imageUrl && <img src={imageUrl} alt="Generated" />}
    </div>
  );
}

export default ImageDisplay;