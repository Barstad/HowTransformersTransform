import React, { useState, useEffect } from 'react';
import axios from 'axios';

interface Token {
  text: string;
  similarity?: number;
}

interface TokenDisplayProps {
  onTokenClick: (index: number) => void;
  selectedLayer: number;
}

const TokenDisplay: React.FC<TokenDisplayProps> = ({ onTokenClick, selectedLayer }) => {
  const [tokens, setTokens] = useState<Token[]>([]);
  const [clickedIndex, setClickedIndex] = useState<number | null>(null);

  const fetchTokenSimilarities = async (tokenIdx: number = 0) => {
    try {
      const response = await axios.post('http://localhost:8000/token_similarities', {
        token_idx: tokenIdx,
        layer_idx: selectedLayer,
        prompt: { text: '' }
      });
      const { tokens: newTokens, similarities } = response.data;
      setTokens(newTokens.map((text: string, i: number) => ({ text, similarity: similarities[i] })));
    } catch (error) {
      console.error('Error fetching token similarities:', error);
    }
  };

  useEffect(() => {
    fetchTokenSimilarities(clickedIndex ?? 0);
  }, [selectedLayer, clickedIndex]);

  const handleTokenClick = (index: number) => {
    setClickedIndex(index);
    onTokenClick(index);
    fetchTokenSimilarities(index);
  };

  const getBackgroundColor = (similarity: number | undefined) => {
    if (similarity === undefined) return 'transparent';
    const hue = similarity * 120;
    return `hsl(${hue}, 100%, 75%)`;
  };

  return (
    <div className="token-display">
      {tokens.map((token, index) => (
        <span
          key={index}
          className={`token ${
            clickedIndex === index ? 'selected' : ''
          }`}
          style={{
            backgroundColor: getBackgroundColor(token.similarity),
          }}
          onClick={() => handleTokenClick(index)}
        >
          {token.text}
        </span>
      ))}
    </div>
  );
};

export default TokenDisplay;