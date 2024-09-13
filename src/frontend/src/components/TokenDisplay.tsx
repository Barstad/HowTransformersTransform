import React, { useState, useEffect } from 'react';
import similarities from '../data/prompt_similarities.json';

// Add this type definition
type SimilarityData = {
  [key in 'small' | 'large']: {
    prompt_tokens: string[];
    layers: {
      [layer: string]: {
        [tokenIndex: string]: {
          similarities: number[];
        };
      };
    };
  };
};

// Cast the imported similarities
const typedSimilarities = similarities as SimilarityData;

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

  useEffect(() => {
    const modelData = typedSimilarities[model];
    const promptTokens = modelData.prompt_tokens;
    const tokenSimilarities = clickedIndex !== null
      ? modelData.layers[selectedLayer.toString()]?.[clickedIndex.toString()]?.similarities ?? []
      : [];

    const newTokens = promptTokens.map((text: string, i: number) => ({
      text,
      similarity: tokenSimilarities[i]
    }));

    setTokens(newTokens);
  }, [selectedLayer, clickedIndex, model]);

  const handleTokenClick = (index: number) => {
    if (index === clickedIndex) {
      setClickedIndex(null);
      onTokenClick(-1);
    } else {
      setClickedIndex(index);
      onTokenClick(index);
    }
  };

  const getBackgroundColor = (similarity: number | undefined) => {
    if (similarity === undefined) return 'transparent';
    
    // Find min and max similarities
    const similarities = tokens.map(token => token.similarity).filter((s): s is number => s !== undefined);
    const minSimilarity = Math.min(...similarities);
    const maxSimilarity = Math.max(...similarities);
    
    // Min/max scaling
    const scaledSimilarity = (similarity - minSimilarity) / (maxSimilarity - minSimilarity);
    
    // Map scaled similarity to hue range [0, 120] (red to green)
    const hue = scaledSimilarity * 120;
    
    return `hsl(${hue}, 100%, 80%)`;
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