import React, { useState, useEffect } from 'react';
import axios from 'axios';

const Predict = () => {
  const [selectedYear, setSelectedYear] = useState('');
  const [predictions, setPredictions] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [warning, setWarning] = useState(null);
  const [activeTab, setActiveTab] = useState('winning');
  const [yearRanges, setYearRanges] = useState(null);

  const tabs = [
    { id: 'winning', label: 'Winning Times'},
    { id: 'a', label: 'A Final Cutoffs'},
    { id: 'b', label: 'B Final Cutoffs'},
    { id: 'c', label: 'C Final Cutoffs'}
  ];

  useEffect(() => {
    fetchYearRanges();
  }, []);

  const fetchYearRanges = async () => {
    try {
      const response = await axios.get('/api/data-status');
      setYearRanges(response.data.year_ranges);
    } catch (err) {
      console.error('Failed to fetch year ranges:', err);
    }
  };

  const formatTime = (seconds) => {
    if (seconds === null || seconds === undefined || isNaN(seconds)) {
      return 'N/A';
    }
    
    if (seconds < 60) {
      return `${seconds.toFixed(2)}s`;
    } else {
      const minutes = Math.floor(seconds / 60);
      const remainingSeconds = seconds % 60;
      return `${minutes}:${remainingSeconds.toFixed(2).padStart(5, '0')}`;
    }
  };

  const getYearWarning = (year) => {
    if (!yearRanges) return null;
    
    const dataRange = yearRanges.data || [2002, 2025];
    const cutoffRange = yearRanges.cutoff_features || [2006, 2025];
    const winningRange = yearRanges.winning_features || [2002, 2025];
    
    const minYear = Math.min(dataRange[0], cutoffRange[0], winningRange[0]);
    const maxYear = Math.max(dataRange[1], cutoffRange[1], winningRange[1]);
    
    if (year < minYear || year > maxYear) {
      return `Warning: Year ${year} is outside the available data range (${minYear}-${maxYear}). Predictions may be less accurate.`;
    }
    
    if (year < cutoffRange[0] || year > cutoffRange[1]) {
      return `Warning: Year ${year} is outside the cutoff features range (${cutoffRange[0]}-${cutoffRange[1]}). Cutoff predictions may be less accurate.`;
    }
    
    return null;
  };

  const generatePredictions = async () => {
    if (!selectedYear) return;
    
    setLoading(true);
    setError(null);
    setWarning(null);
    
    const year = parseInt(selectedYear);
    const yearWarning = getYearWarning(year);
    if (yearWarning) {
      setWarning(yearWarning);
    }
    
    try {
      const response = await axios.get(`/api/predict-simple/${year}`);
      
      if (response.data.status === 'success') {
        console.log('Predictions response:', response.data);
        setPredictions(response.data.predictions);
      } else {
        setError('Failed to generate predictions');
      }
    } catch (err) {
      setError('Failed to generate predictions');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const renderTable = () => {
    if (!predictions) return null;
    
    console.log('Rendering table with predictions:', predictions);

    const getColumnData = () => {
      switch (activeTab) {
        case 'winning':
          return predictions.map(event => ({
            event: event.event,
            predicted: event.predictions.simple_winning,
            actual: event.predictions.actual?.winning_time,
            difference: event.predictions.actual?.winning_time ? 
              event.predictions.simple_winning - event.predictions.actual.winning_time : null
          }));
        case 'a':
          return predictions.map(event => ({
            event: event.event,
            predicted: event.predictions.simple_cutoff?.A,
            actual: event.predictions.actual?.a_cutoff,
            difference: event.predictions.actual?.a_cutoff ? 
              event.predictions.simple_cutoff?.A - event.predictions.actual.a_cutoff : null
          }));
        case 'b':
          return predictions.map(event => ({
            event: event.event,
            predicted: event.predictions.simple_cutoff?.B,
            actual: event.predictions.actual?.b_cutoff,
            difference: event.predictions.actual?.b_cutoff ? 
              event.predictions.simple_cutoff?.B - event.predictions.actual.b_cutoff : null
          }));
        case 'c':
          return predictions.map(event => ({
            event: event.event,
            predicted: event.predictions.simple_cutoff?.C,
            actual: event.predictions.actual?.c_cutoff,
            difference: event.predictions.actual?.c_cutoff ? 
              event.predictions.simple_cutoff?.C - event.predictions.actual.c_cutoff : null
          }));
        default:
          return [];
      }
    };

    const tableData = getColumnData();
    
    console.log('Table data for tab', activeTab, ':', tableData);

    return (
      <div className="overflow-x-auto">
        <table className="min-w-full bg-white border border-gray-300">
          <thead>
            <tr className="bg-gray-50">
              <th className="px-6 py-3 border-b border-gray-300 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Event
              </th>
              <th className="px-6 py-3 border-b border-gray-300 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Predicted Time
              </th>
              <th className="px-6 py-3 border-b border-gray-300 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Actual Time
              </th>
              <th className="px-6 py-3 border-b border-gray-300 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Difference
              </th>
            </tr>
          </thead>
          <tbody className="bg-white divide-y divide-gray-300">
            {tableData.map((row, index) => (
              <tr key={index} className="hover:bg-gray-50">
                <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                  {row.event}
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                  {formatTime(row.predicted)}
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                  {formatTime(row.actual)}
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-sm">
                  {row.difference !== null ? (
                    <span className={`font-medium ${row.difference > 0 ? 'text-red-600' : 'text-green-600'}`}>
                      {row.difference > 0 ? '+' : ''}{formatTime(row.difference)}
                    </span>
                  ) : (
                    <span className="text-gray-400">N/A</span>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    );
  };

  return (
    <div className="space-y-6">
      <div className="bg-white rounded-lg shadow p-6">
        <div className="flex justify-between items-center mb-6">
          <h1 className="text-3xl font-bold text-gray-900">Predict Times</h1>
          <div className="flex items-center space-x-4">
            <label className="text-sm font-medium text-gray-700">Year:</label>
            <input
              type="number"
              value={selectedYear}
              onChange={(e) => setSelectedYear(e.target.value)}
              placeholder="Enter year (e.g., 2025)"
              className="border border-gray-300 rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
              min="1990"
              max="2030"
            />
            <button
              onClick={generatePredictions}
              disabled={!selectedYear || loading}
              className="bg-blue-600 text-white px-4 py-2 rounded-md hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed"
            >
              {loading ? 'Generating...' : 'Generate Predictions'}
            </button>
          </div>
        </div>

        {warning && (
          <div className="bg-yellow-50 border border-yellow-200 rounded-md p-4 mb-6">
            <p className="text-yellow-700">{warning}</p>
          </div>
        )}

        {error && (
          <div className="bg-red-50 border border-red-200 rounded-md p-4 mb-6">
            <p className="text-red-700">{error}</p>
          </div>
        )}

        {loading && (
          <div className="flex justify-center items-center h-64">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
          </div>
        )}

        {predictions && !loading && (
          <div className="space-y-6">
            <div className="border-b border-gray-200">
              <nav className="-mb-px flex space-x-8">
                {tabs.map((tab) => (
                  <button
                    key={tab.id}
                    onClick={() => setActiveTab(tab.id)}
                    className={`py-2 px-1 border-b-2 font-medium text-sm ${
                      activeTab === tab.id
                        ? 'border-blue-500 text-blue-600'
                        : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                    }`}
                  >
                    <span className="mr-2">{tab.icon}</span>
                    {tab.label}
                  </button>
                ))}
              </nav>
            </div>

            <div className="mt-6">
              <h2 className="text-xl font-semibold text-gray-900 mb-4">
                {tabs.find(tab => tab.id === activeTab)?.label} for {selectedYear}
              </h2>
              {renderTable()}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default Predict;
