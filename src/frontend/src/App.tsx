import React, { useState } from 'react';
import TokenDisplay from './components/TokenDisplay';
import WordCloud from './components/WordCloud';
import './App.css'; // Add this import for custom styles

const App: React.FC = () => {
  const [selectedTokenIndex, setSelectedTokenIndex] = useState<number>(0);
  // You'll need to fetch or generate this data
  const tokenVectors: Array<[number, number]> = [/* Your token vector data here */];

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
      </main>
    </div>
  );
};

export default App;