import React, { useState, useEffect } from 'react';
import axios from 'axios';

interface Token {
  text: string;
  similarity?: number;
}

const TokenDisplay: React.FC = () => {
  const [tokens, setTokens] = useState<Token[]>([]);
  const [clickedIndex, setClickedIndex] = useState<number>(0);

  useEffect(() => {
    fetchTokensAndSimilarities();
  }, []);

  const fetchTokensAndSimilarities = async () => {
    try {
      const response = await axios.post('http://localhost:8000/token_similarities', {
        token_idx: 0,
        prompt: { text: '' }
      });
      const { tokens, similarities } = response.data;
      setTokens(tokens.map((text: string, i: number) => ({ text, similarity: similarities[i] })));
    } catch (error) {
      console.error('Error fetching tokens and similarities:', error);
    }
  };

  const handleTokenClick = async (token: Token, index: number) => {
    console.log('Token clicked:', token.text);
    console.log('Token length:', token.text.length);
    console.log('Token index:', index);
    setClickedIndex(index);
    try {
      const response = await axios.post('http://localhost:8000/token_similarities', {
        token_idx: index,
        prompt: { text: '' }
      });
      const similarities = response.data.similarities;
      setTokens(prevTokens => 
        prevTokens.map((token, i) => ({ ...token, similarity: similarities[i] }))
      );
    } catch (error) {
      console.error('Error fetching similarities:', error);
    }
  };

  const getBackgroundColor = (similarity: number | undefined) => {
    if (similarity === undefined) return 'transparent';
    // Red: hsl(0, 100%, 50%), Green: hsl(120, 100%, 50%)
    const hue = similarity * 120; // 0 (red) to 120 (green)
    const saturation = 100;
    const lightness = 75; // Increased from 75 to 85 for better contrast
    return `hsl(${hue}, ${saturation}%, ${lightness}%)`;
  };

  return (
    <div className="flex justify-center items-center min-h-screen bg-gray-100">
      <div className="bg-white p-8 rounded-lg shadow-md max-w-3xl w-full">
        <div className="flex flex-wrap">
          {tokens.map((token, index) => (
            <span
              key={index}
              className={`inline-block whitespace-pre cursor-pointer transition-colors duration-200 ${
                clickedIndex === null || clickedIndex === index
                  ? 'hover:bg-gray-200'
                  : 'text-gray-600 hover:text-gray-800'
              }`}
              style={{
                fontFamily: '"Roboto", sans-serif',
                fontSize: '16px',
                color: clickedIndex === null || clickedIndex === index ? 'hsl(240, 10%, 3.9%)' : undefined,
                backgroundColor: getBackgroundColor(token.similarity),
              }}
              onClick={() => handleTokenClick(token, index)}
            >
              {token.text}
            </span>
          ))}
        </div>
      </div>
    </div>
  );
};

export default TokenDisplay;