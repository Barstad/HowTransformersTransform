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
        <h1 className="text-2xl font-bold">Token Analysis Dashboard</h1>
      </header>
      <main className="flex-grow flex flex-col p-4 gap-4">
        <section className="token-display">
          <h2 className="text-xl mb-2">Token Display</h2>
          <TokenDisplay onTokenClick={handleTokenClick} />
        </section>
        <div className="plots-container flex-grow flex flex-col sm:flex-row gap-4">
          <section className="word-cloud flex-1">
            <h2 className="text-xl mb-2">Word Cloud</h2>
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