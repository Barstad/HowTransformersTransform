import React, { useState } from 'react';
import ReactMarkdown from 'react-markdown';
import TokenDisplay from './components/TokenDisplay';
import WordCloud from './components/WordCloud';
import Plot2dPointsD3 from './components/Plot2dPointsD3';
import LayerMenu from './components/SelectMenu';
import './App.css'; // Add this import for custom styles

const App: React.FC = () => {
  const [selectedTokenIndex, setSelectedTokenIndex] = useState<number>(0);
  const [selectedLayer, setSelectedLayer] = useState<number>(0);
  const [selectedModel, setSelectedModel] = useState<"small" | "large">("small");

  const handleTokenClick = (index: number) => {
    setSelectedTokenIndex(index);
  };

  const handleLayerSelect = (index: number) => {
    setSelectedLayer(index);
  };

  const handleModelSelect = (event: React.ChangeEvent<HTMLSelectElement>) => {
    setSelectedModel(event.target.value as "small" | "large");
  };

  const description = `
Transformer models are mysterious things. A token mapped to an integer goes in, gets projected into a high dimensional space, and is then subsequently shaped by its surroundings. We want to take a closer look at this journey. 

This tool explores how tokens transform as they move through the layers of a transformer model. The approach is simple, try to visualize how tokens relate by leveraging that they keep to the same embedding space. Let's explore how relations between tokens changes as they are transformed. They start out as an entry in an embedding matrix, and are gradually transformed into something else. This is fascinating, and hopefully this tool can help build some intuition.
  `;

  // TODO: Make plotd2points work with layer_idx

  return (
    <div className="app-container min-h-screen flex flex-col">
      <header className="p-4">
        <h1 className="text-2xl font-bold">How Transformers Transform</h1>
      </header>
      <ReactMarkdown className="p-description flex flex-col p-4 gap-4">
        {description}
      </ReactMarkdown>
      <main className="flex-grow flex flex-col p-4 gap-4">
        <div className="flex justify-between items-center">
          <div className="layer-menu">
            <h2 className="text-xl mb-2">Select Layer</h2>
            <LayerMenu onLayerSelect={handleLayerSelect} model={selectedModel} />
          </div>
          <div className="model-select">
            <h2 className="text-xl mb-2">Select Model</h2>
            <select
              value={selectedModel}
              onChange={handleModelSelect}
              className="p-2 border rounded"
            >
              <option value="small">Small</option>
              <option value="large">Large</option>
            </select>
          </div>
        </div>
        <div className="token-display">
          <h2 className="token-display-header">A tokenized story. Try to click a word.</h2>
          <TokenDisplay
            onTokenClick={handleTokenClick}
            selectedLayer={selectedLayer}
            model={selectedModel}
          />
        </div>
        <div className="plots-container flex-grow flex flex-col sm:flex-row gap-4">
          <div className="word-cloud flex-1">
            <h2 className="word-cloud-header">Most similar tokens</h2>
            <div className="h-full">
              <WordCloud
                selectedTokenIndex={selectedTokenIndex}
                selectedLayer={selectedLayer}
                model={selectedModel}
              />
            </div>
          </div>
          <div className="embeddings-plot flex-1">
            <h2 className="plot-2d-header">2D Projections</h2>
            {/* <Plot2dPointsD3 layer_idx={selectedLayer} model={selectedModel} /> */}
          </div>
        </div>
      </main>
    </div>
  );
};

export default App;