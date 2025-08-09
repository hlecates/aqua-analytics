import React, { useState } from 'react';
import { User, Clock, Trophy } from 'lucide-react';
import axios from 'axios';

const PersonalAnalysis = () => {
  const [personalTimes, setPersonalTimes] = useState({});
  const [analysis, setAnalysis] = useState(null);
  const [loading, setLoading] = useState(false);
  const [selectedYear, setSelectedYear] = useState(2025);

  const events = [
    'Men 50 Freestyle',
    'Men 100 Freestyle', 
    'Men 200 Freestyle',
    'Men 500 Freestyle',
    'Men 50 Backstroke',
    'Men 100 Backstroke',
    'Men 200 Backstroke',
    'Men 50 Breaststroke',
    'Men 100 Breaststroke',
    'Men 200 Breaststroke',
    'Men 50 Butterfly',
    'Men 100 Butterfly',
    'Men 200 Butterfly',
    'Men 200 IM',
    'Men 400 IM'
  ];

  const handleTimeChange = (event, time) => {
    setPersonalTimes(prev => ({
      ...prev,
      [event]: parseFloat(time) || 0
    }));
  };

  const analyzeTimes = async () => {
    setLoading(true);
    
    try {
      const response = await axios.post('/api/personal-analysis', {
        year: selectedYear,
        personal_times: personalTimes
      });
      
      setAnalysis(response.data);
    } catch (error) {
      console.error('Analysis failed:', error);
    } finally {
      setLoading(false);
    }
  };

  const formatTime = (seconds) => {
    if (seconds < 60) {
      return `${seconds.toFixed(2)}s`;
    } else {
      const minutes = Math.floor(seconds / 60);
      const remainingSeconds = seconds % 60;
      return `${minutes}:${remainingSeconds.toFixed(2).padStart(5, '0')}`;
    }
  };

  return (
    <div className="space-y-6">
      <div className="bg-white rounded-lg shadow p-6">
        <h1 className="text-3xl font-bold text-gray-900 mb-6">Personal Time Analysis</h1>
        
        <div className="mb-6">
          <label className="block text-sm font-medium text-gray-700 mb-2">Year:</label>
          <select
            value={selectedYear}
            onChange={(e) => setSelectedYear(parseInt(e.target.value))}
            className="border border-gray-300 rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            <option value={2024}>2024</option>
            <option value={2025}>2025</option>
          </select>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mb-6">
          {events.map((event) => (
            <div key={event} className="space-y-2">
              <label className="block text-sm font-medium text-gray-700">{event}</label>
              <input
                type="number"
                step="0.01"
                placeholder="Enter time (seconds)"
                value={personalTimes[event] || ''}
                onChange={(e) => handleTimeChange(event, e.target.value)}
                className="w-full border border-gray-300 rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
          ))}
        </div>

        <button
          onClick={analyzeTimes}
          disabled={loading}
          className="bg-blue-600 text-white px-6 py-2 rounded-md hover:bg-blue-700 disabled:opacity-50 transition-colors"
        >
          {loading ? 'Analyzing...' : 'Analyze Times'}
        </button>
      </div>

      {analysis && (
        <div className="bg-white rounded-lg shadow p-6">
          <h2 className="text-2xl font-semibold text-gray-900 mb-6">Analysis Results</h2>
          
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {analysis.analysis.map((result) => (
              <div key={result.event} className="border border-gray-200 rounded-lg p-4">
                <h3 className="text-lg font-semibold text-gray-900 mb-3">{result.event}</h3>
                
                <div className="space-y-3">
                  <div className="flex items-center space-x-2">
                    <Clock className="h-4 w-4 text-gray-500" />
                    <span className="text-sm text-gray-600">Your Time:</span>
                    <span className="font-medium">{formatTime(result.personal_time)}</span>
                  </div>
                  
                  <div className="flex items-center space-x-2">
                    <Trophy className="h-4 w-4 text-green-500" />
                    <span className="text-sm text-gray-600">Simple Model Finals:</span>
                    <span className="font-medium">
                      {result.simple_finals.length > 0 ? result.simple_finals.join(', ') : 'None'}
                    </span>
                  </div>
                  
                  <div className="flex items-center space-x-2">
                    <User className="h-4 w-4 text-blue-500" />
                    <span className="text-sm text-gray-600">Advanced Model Finals:</span>
                    <span className="font-medium">
                      {result.advanced_finals.length > 0 ? result.advanced_finals.join(', ') : 'None'}
                    </span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default PersonalAnalysis; 