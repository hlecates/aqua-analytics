import React, { useState, useEffect } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import axios from 'axios';

const Predictions = () => {
  const [predictions, setPredictions] = useState([]);
  const [selectedYear, setSelectedYear] = useState(2025);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const years = [2024, 2025];

  const fetchPredictions = async (year) => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await axios.get(`/api/predictions/${year}`);
      setPredictions(response.data.predictions);
    } catch (err) {
      setError('Failed to fetch predictions');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchPredictions(selectedYear);
  }, [selectedYear]);

  const formatTime = (seconds) => {
    if (seconds < 60) {
      return `${seconds.toFixed(2)}s`;
    } else {
      const minutes = Math.floor(seconds / 60);
      const remainingSeconds = seconds % 60;
      return `${minutes}:${remainingSeconds.toFixed(2).padStart(5, '0')}`;
    }
  };

  if (loading) {
    return (
      <div className="flex justify-center items-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="bg-white rounded-lg shadow p-6">
        <div className="flex justify-between items-center mb-6">
          <h1 className="text-3xl font-bold text-gray-900">Predictions</h1>
          <div className="flex items-center space-x-4">
            <label className="text-sm font-medium text-gray-700">Year:</label>
            <select
              value={selectedYear}
              onChange={(e) => setSelectedYear(parseInt(e.target.value))}
              className="border border-gray-300 rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              {years.map(year => (
                <option key={year} value={year}>{year}</option>
              ))}
            </select>
          </div>
        </div>

        {error && (
          <div className="bg-red-50 border border-red-200 rounded-md p-4 mb-6">
            <p className="text-red-700">{error}</p>
          </div>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {predictions.map((event) => (
            <div key={event.event} className="bg-gray-50 rounded-lg p-6">
              <h3 className="text-xl font-semibold text-gray-900 mb-4">{event.event}</h3>
              
              <div className="space-y-4">
                <div>
                  <h4 className="font-medium text-gray-700 mb-2">Winning Times</h4>
                  <div className="grid grid-cols-2 gap-2 text-sm">
                    <div>
                      <span className="text-gray-500">Simple Model:</span>
                      <span className="ml-2 font-medium">
                        {formatTime(event.predictions.simple_winning)}
                      </span>
                    </div>
                    <div>
                      <span className="text-gray-500">Advanced Model:</span>
                      <span className="ml-2 font-medium">
                        {formatTime(event.predictions.advanced_winning)}
                      </span>
                    </div>
                  </div>
                </div>

                <div>
                  <h4 className="font-medium text-gray-700 mb-2">Cutoff Times</h4>
                  <div className="space-y-2">
                    {['A', 'B', 'C'].map((final) => (
                      <div key={final} className="grid grid-cols-3 gap-2 text-sm">
                        <span className="text-gray-500">{final} Final:</span>
                        <span className="font-medium">
                          {formatTime(event.predictions.simple_cutoff[final])}
                        </span>
                        <span className="font-medium">
                          {formatTime(event.predictions.advanced_cutoff[final])}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default Predictions; 