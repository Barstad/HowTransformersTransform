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
    <div className="app-container">
      <header className="app-header">
        <h1 className="main-title">Token Analysis Dashboard</h1>
      </header>
      <main className="app-main">
        <section className="dashboard-section token-display-section">
          <h2 className="section-title">Token Display</h2>
          <TokenDisplay onTokenClick={handleTokenClick} />
        </section>
        <section className="dashboard-section word-cloud-section">
          <h2 className="section-title">Word Cloud</h2>
          <WordCloud 
            selectedTokenIndex={selectedTokenIndex} 
            width={800} 
            height={600} 
          />
        </section>
        <section className="dashboard-section plot-section">
          <h2 className="section-title">2D Embeddings Plot</h2>
          <Plot2dPointsD3 />
        </section>
      </main>
    </div>
  );
};

export default App;