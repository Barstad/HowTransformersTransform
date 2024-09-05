import React from 'react';
import TokenDisplay from './components/TokenDisplay';

function App() {
  return (
    <div className="App">
      <header className="bg-black text-white p-4">
        <h1 className="text-2xl font-bold text-center font-roboto">Transformer Visualization</h1>
      </header>
      <main className="p-4">
        <TokenDisplay />
      </main>
    </div>
  );
}

export default App;