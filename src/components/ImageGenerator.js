import React, { useState } from 'react';

function ImageGenerator({ onGenerate }) {
  const [query, setQuery] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    onGenerate(query);
  };

  return (
    <form onSubmit={handleSubmit}>
      <input
        type="text"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        placeholder="Enter image description"
      />
      <button type="submit">Generate Image</button>
    </form>
  );
}

export default ImageGenerator;