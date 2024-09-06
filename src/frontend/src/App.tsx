import React, { useState } from 'react';
import TokenDisplay from './components/TokenDisplay';
import WordCloud from './components/WordCloud';
import Plot2dPointsD3 from './components/Plot2dPointsD3';
import './App.css'; // Add this import for custom styles

const App: React.FC = () => {
  const [selectedTokenIndex, setSelectedTokenIndex] = useState<number>(0);

  const handleTokenClick = (index: number) => {
    setSelectedTokenIndex(index);
  };

  return (
    <div className="app-container min-h-screen flex flex-col">
      <header className="p-4">
        <h1 className="text-2xl font-bold">How Transformers Transform</h1>
      </header>
      <p className="p-description flex flex-col p-4 gap-4">
        Transformer models are mysterious things. A token goes through a machine, starting in one location, and ends up in another. This is an attempt to understand this journey a bit better. This tool explores how tokens transform as they move through the layers of a transformer model. The approach is simple, try to visualize how tokens relate by leveraging that they keep to the same embedding space. Lets explore how relations between tokens changes as they are transformed. They start out as en entry in an embedding matrix, and are gradually transformed into something else. This is fascinating, and hopefully this tool can help build some intuition.
      </p>
      <main className="flex-grow flex flex-col p-4 gap-4">
        <section className="token-display">
          <h2 className="text-xl mb-2">Token Display</h2>
          <TokenDisplay onTokenClick={handleTokenClick} />
        </section>
        <div className="plots-container flex-grow flex flex-col sm:flex-row gap-4">
          <section className="word-cloud flex-1">
            <h2 className="text-xl mb-2">Most similar tokens</h2>
            <div className="h-full">
              <WordCloud selectedTokenIndex={selectedTokenIndex} />
            </div>
          </section>
          <section className="embeddings-plot flex-1">
            <h2 className="text-xl mb-2">2D Projections</h2>
            <Plot2dPointsD3 />
          </section>
        </div>
      </main>
    </div>
  );
};

export default App;