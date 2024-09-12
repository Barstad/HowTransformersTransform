import React, { useState, useEffect } from 'react';
import axios from 'axios';

interface Token {
  text: string;
  similarity?: number;
}

interface TokenDisplayProps {
  onTokenClick: (index: number) => void;
  selectedLayer: number;
  model: 'small' | 'large';
}

const TokenDisplay: React.FC<TokenDisplayProps> = ({ onTokenClick, selectedLayer, model }) => {
  const [tokens, setTokens] = useState<Token[]>([]);
  const [clickedIndex, setClickedIndex] = useState<number | null>(null);

  const fetchTokenSimilarities = async (tokenIdx: number = 0) => {
    try {
      const response = await axios.post('http://localhost:8000/token_similarities', {
        token_idx: tokenIdx,
        layer_idx: selectedLayer,
        prompt: { text: '' },
        model: model
      });
      const { tokens: newTokens, similarities } = response.data;
      setTokens(newTokens.map((text: string, i: number) => ({ text, similarity: similarities[i] })));
    } catch (error) {
      console.error('Error fetching token similarities:', error);
    }
  };

  useEffect(() => {
    fetchTokenSimilarities(clickedIndex !== null ? clickedIndex : 0);
  }, [selectedLayer, clickedIndex]);

  const handleTokenClick = (index: number) => {
    if (index === clickedIndex) {
      // Deselect the token if it's already selected
      setClickedIndex(null);
      onTokenClick(-1); // Use -1 to indicate no token is selected
    } else {
      setClickedIndex(index);
      onTokenClick(index);
    }
    fetchTokenSimilarities(index);
  };

  const getBackgroundColor = (similarity: number | undefined) => {
    if (similarity === undefined) return 'transparent';
    const hue = similarity * 120;
    return `hsl(${hue}, 100%, 75%)`;
  };

  return (
    <div>
      <div className="token-display-container">
        {tokens.map((token, index) => (
          <span
            key={index}
            className={`token ${
              clickedIndex === index ? 'selected' : ''
            }`}
            style={{
              backgroundColor: clickedIndex !== null ? getBackgroundColor(token.similarity) : 'transparent',
            }}
            onClick={() => handleTokenClick(index)}
          >
            {token.text}
          </span>
        ))}
      </div>
      <div className="active-token-display">
        <p className="active-token-display-text">
          {clickedIndex !== null ? tokens[clickedIndex]?.text : 'Click a token to see how the transformer sees it.'}
        </p>
      </div>
    </div>
  );
};

export default TokenDisplay;