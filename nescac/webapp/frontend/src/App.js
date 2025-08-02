import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Navbar from './components/Navbar';
import Dashboard from './pages/Dashboard';
import Predictions from './pages/Predictions';
import PersonalAnalysis from './pages/PersonalAnalysis';
import HistoricalData from './pages/HistoricalData';
import './App.css';

function App() {
  return (
    <Router>
      <div className="min-h-screen bg-gray-50">
        <Navbar />
        <main className="container mx-auto px-4 py-8">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/predictions" element={<Predictions />} />
            <Route path="/personal-analysis" element={<PersonalAnalysis />} />
            <Route path="/historical-data" element={<HistoricalData />} />
          </Routes>
        </main>
      </div>
    </Router>
  );
}

export default App; 