import React, { useEffect, useMemo, useState } from 'react';
import axios from 'axios';
import { Search } from 'lucide-react';

const formatTime = (sec) => {
  if (sec === null || sec === undefined || Number.isNaN(sec)) return '—';
  const total = Math.max(0, Number(sec));
  let minutes = Math.floor(total / 60);
  let rem = total - minutes * 60;
  let seconds = Math.floor(rem);
  let hundredths = Math.round((rem - seconds) * 100);
  if (hundredths === 100) { seconds += 1; hundredths = 0; }
  if (seconds === 60) { minutes += 1; seconds = 0; }
  const mm = String(minutes).padStart(2, '0');
  const ss = String(seconds).padStart(2, '0');
  const hh = String(hundredths).padStart(2, '0');
  return `${mm}:${ss}.${hh}`;
};

const Historical = () => {
  const [mode, setMode] = useState(''); // '', 'year', 'event', 'athlete'

  // Shared options
  const [years, setYears] = useState([]);
  const [allEvents, setAllEvents] = useState([]);

  // Year-first flow
  const [yearModeYear, setYearModeYear] = useState('');
  const [yearModeEvents, setYearModeEvents] = useState([]);
  const [yearModeEventId, setYearModeEventId] = useState('');

  // Event-first flow
  const [eventModeEventId, setEventModeEventId] = useState('');
  const [eventModeYears, setEventModeYears] = useState([]);
  const [eventModeYear, setEventModeYear] = useState('');

  // Results
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Athlete search
  const [athleteQuery, setAthleteQuery] = useState('');
  const [athleteResults, setAthleteResults] = useState([]);

  useEffect(() => {
    const boot = async () => {
      try {
        const yrsRes = await axios.get('/api/history/years');
        setYears(yrsRes.data.years || []);
        const evRes = await axios.get('/api/history/events', { params: { include_relay: false } });
        setAllEvents(evRes.data.events || []);
      } catch (e) {
        // eslint-disable-next-line no-console
        console.error(e);
      }
    };
    boot();
  }, []);

  const resetAll = () => {
    setYearModeYear('');
    setYearModeEvents([]);
    setYearModeEventId('');
    setEventModeEventId('');
    setEventModeYears([]);
    setEventModeYear('');
    setResults(null);
    setError(null);
    setLoading(false);
    setAthleteQuery('');
    setAthleteResults([]);
  };

  const onChangeMode = (m) => {
    setMode(m);
    resetAll();
  };

  const loadEventsForYear = async (year) => {
    try {
      const res = await axios.get('/api/history/events', { params: { year, include_relay: false } });
      setYearModeEvents(res.data.events || []);
    } catch (e) {
      // eslint-disable-next-line no-console
      console.error(e);
      setYearModeEvents([]);
    }
  };

  const loadYearsForEvent = async (evId) => {
    try {
      const res = await axios.get('/api/history/event-years', { params: { event_id: evId } });
      setEventModeYears(res.data.years || []);
    } catch (e) {
      // eslint-disable-next-line no-console
      console.error(e);
      setEventModeYears([]);
    }
  };

  const fetchResults = async (evId, year) => {
    if (!evId || !year) return;
    setLoading(true);
    setError(null);
    setResults(null);
    try {
      const res = await axios.get('/api/history/results', { params: { event_id: evId, year } });
      setResults(res.data);
    } catch (e) {
      setError('Failed to fetch results');
    } finally {
      setLoading(false);
    }
  };

  const searchAthlete = async () => {
    if (!athleteQuery.trim()) { setAthleteResults([]); return; }
    setLoading(true);
    setError(null);
    try {
      const res = await axios.get('/api/history/athlete', { params: { q: athleteQuery } });
      setAthleteResults(res.data.results || []);
    } catch (e) {
      setError('Failed to search athlete');
    } finally {
      setLoading(false);
    }
  };

  const groupedAthlete = useMemo(() => {
    const map = {};
    for (const r of athleteResults) {
      const y = r.year || 'Unknown';
      if (!map[y]) map[y] = [];
      map[y].push(r);
    }
    const yearsSorted = Object.keys(map).filter((k) => k !== 'Unknown').map((k) => Number(k)).sort((a, b) => a - b);
    const unknown = map['Unknown'] || [];
    const ordered = [...yearsSorted.map((y) => ({ year: y, rows: map[y] })), ...(unknown.length ? [{ year: 'Unknown', rows: unknown }] : [])];
    return ordered;
  }, [athleteResults]);

  return (
    <div className="space-y-6">
      <div className="bg-white rounded-lg shadow p-6">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">Historical Results</h1>
        <p className="text-sm text-gray-600">Choose how you want to explore: by year, by event, or by athlete.</p>
      </div>

      <div className="bg-white rounded-lg shadow p-6 space-y-6">
        <div className="flex items-center space-x-6">
          <label className="inline-flex items-center space-x-2">
            <input type="radio" name="mode" className="h-4 w-4" value="year" checked={mode === 'year'} onChange={() => onChangeMode('year')} />
            <span className="text-gray-800 font-medium">By Year</span>
          </label>
          <label className="inline-flex items-center space-x-2">
            <input type="radio" name="mode" className="h-4 w-4" value="event" checked={mode === 'event'} onChange={() => onChangeMode('event')} />
            <span className="text-gray-800 font-medium">By Event</span>
          </label>
          <label className="inline-flex items-center space-x-2">
            <input type="radio" name="mode" className="h-4 w-4" value="athlete" checked={mode === 'athlete'} onChange={() => onChangeMode('athlete')} />
            <span className="text-gray-800 font-medium">By Athlete</span>
          </label>
        </div>

        {mode === '' && (
          <div className="text-sm text-gray-600">Select a mode above to get started.</div>
        )}

        {mode === 'year' && (
          <div className="space-y-4">
            <div className="flex items-center gap-3">
              <label className="text-sm font-medium text-gray-700">Year</label>
              <select
                value={yearModeYear}
                onChange={async (e) => {
                  const y = e.target.value;
                  setYearModeYear(y);
                  setYearModeEventId('');
                  setResults(null);
                  if (y) await loadEventsForYear(Number(y));
                }}
                className="border border-gray-300 rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="">Select year</option>
                {years.map((y) => (
                  <option key={y} value={y}>{y}</option>
                ))}
              </select>
            </div>

            {yearModeYear && (
              <div className="flex items-center gap-3">
                <label className="text-sm font-medium text-gray-700">Event</label>
                <select
                  value={yearModeEventId}
                  onChange={async (e) => {
                    const evId = e.target.value;
                    setYearModeEventId(evId);
                    setResults(null);
                    if (evId) await fetchResults(Number(evId), Number(yearModeYear));
                  }}
                  className="border border-gray-300 rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                  <option value="">Select event</option>
                  {yearModeEvents.map((ev) => (
                    <option key={ev.id} value={ev.id}>{ev.name}</option>
                  ))}
                </select>
              </div>
            )}

            {loading && <div className="text-sm text-gray-500">Loading…</div>}
            {error && <div className="bg-red-50 border border-red-200 rounded-md p-4 text-red-700">{error}</div>}

            {results && (
              <div className="space-y-4">
                <h3 className="font-semibold text-gray-900">{results.year} — {results.event?.name}</h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="bg-gray-50 rounded p-4">
                    <h5 className="font-medium text-gray-700 mb-2">Finals</h5>
                    <div className="space-y-1">
                      {results.finals.map((r) => (
                        <div key={r.id} className="flex justify-between text-sm">
                          <span className="text-gray-800">{r.place ? `${r.place}. ` : ''}{r.athlete_name} {r.school_name ? `(${r.school_name})` : ''}</span>
                          <span className="font-semibold">{r.time_raw || formatTime(r.time_seconds)}</span>
                        </div>
                      ))}
                      {results.finals.length === 0 && <div className="text-xs text-gray-500">No finals recorded</div>}
                    </div>
                  </div>
                  <div className="bg-gray-50 rounded p-4">
                    <h5 className="font-medium text-gray-700 mb-2">Prelims</h5>
                    <div className="space-y-1">
                      {results.prelims.map((r) => (
                        <div key={r.id} className="flex justify-between text-sm">
                          <span className="text-gray-800">{r.place ? `${r.place}. ` : ''}{r.athlete_name} {r.school_name ? `(${r.school_name})` : ''}</span>
                          <span className="font-semibold">{r.time_raw || formatTime(r.time_seconds)}</span>
                        </div>
                      ))}
                      {results.prelims.length === 0 && <div className="text-xs text-gray-500">No prelims recorded</div>}
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {mode === 'event' && (
          <div className="space-y-4">
            <div className="flex items-center gap-3">
              <label className="text-sm font-medium text-gray-700">Event</label>
              <select
                value={eventModeEventId}
                onChange={async (e) => {
                  const evId = e.target.value;
                  setEventModeEventId(evId);
                  setEventModeYear('');
                  setResults(null);
                  if (evId) await loadYearsForEvent(Number(evId));
                }}
                className="border border-gray-300 rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="">Select event</option>
                {allEvents.map((ev) => (
                  <option key={ev.id} value={ev.id}>{ev.name}</option>
                ))}
              </select>
            </div>

            {eventModeEventId && (
              <div className="flex items-center gap-3">
                <label className="text-sm font-medium text-gray-700">Year</label>
                <select
                  value={eventModeYear}
                  onChange={async (e) => {
                    const y = e.target.value;
                    setEventModeYear(y);
                    setResults(null);
                    if (y) await fetchResults(Number(eventModeEventId), Number(y));
                  }}
                  className="border border-gray-300 rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                  <option value="">Select year</option>
                  {eventModeYears.map((y) => (
                    <option key={y} value={y}>{y}</option>
                  ))}
                </select>
              </div>
            )}

            {loading && <div className="text-sm text-gray-500">Loading…</div>}
            {error && <div className="bg-red-50 border border-red-200 rounded-md p-4 text-red-700">{error}</div>}

            {results && (
              <div className="space-y-4">
                <h3 className="font-semibold text-gray-900">{results.year} — {results.event?.name}</h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="bg-gray-50 rounded p-4">
                    <h5 className="font-medium text-gray-700 mb-2">Finals</h5>
                    <div className="space-y-1">
                      {results.finals.map((r) => (
                        <div key={r.id} className="flex justify-between text-sm">
                          <span className="text-gray-800">{r.place ? `${r.place}. ` : ''}{r.athlete_name} {r.school_name ? `(${r.school_name})` : ''}</span>
                          <span className="font-semibold">{r.time_raw || formatTime(r.time_seconds)}</span>
                        </div>
                      ))}
                      {results.finals.length === 0 && <div className="text-xs text-gray-500">No finals recorded</div>}
                    </div>
                  </div>
                  <div className="bg-gray-50 rounded p-4">
                    <h5 className="font-medium text-gray-700 mb-2">Prelims</h5>
                    <div className="space-y-1">
                      {results.prelims.map((r) => (
                        <div key={r.id} className="flex justify-between text-sm">
                          <span className="text-gray-800">{r.place ? `${r.place}. ` : ''}{r.athlete_name} {r.school_name ? `(${r.school_name})` : ''}</span>
                          <span className="font-semibold">{r.time_raw || formatTime(r.time_seconds)}</span>
                        </div>
                      ))}
                      {results.prelims.length === 0 && <div className="text-xs text-gray-500">No prelims recorded</div>}
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {mode === 'athlete' && (
          <div className="space-y-6">
            <div className="flex items-center gap-2">
              <div className="relative flex-1">
                <input
                  type="text"
                  placeholder="Search athlete by name"
                  value={athleteQuery}
                  onChange={(e) => setAthleteQuery(e.target.value)}
                  className="w-full border rounded-md pl-10 pr-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
                <Search className="h-4 w-4 text-gray-400 absolute left-3 top-1/2 -translate-y-1/2" />
              </div>
              <button onClick={searchAthlete} className="px-4 py-2 rounded bg-blue-600 text-white hover:bg-blue-700">Search</button>
            </div>
            {athleteResults.length === 0 && athleteQuery && !loading && !error && (
              <div className="text-sm text-gray-500">No results found</div>
            )}
            {error && <div className="bg-red-50 border border-red-200 rounded-md p-4 text-red-700">{error}</div>}
            {loading && <div className="text-sm text-gray-500">Loading…</div>}
            {groupedAthlete.length > 0 && (
              <div className="space-y-4">
                {groupedAthlete.map((g) => (
                  <div key={g.year} className="bg-gray-50 rounded p-4">
                    <h4 className="font-semibold text-gray-800 mb-2">{g.year}</h4>
                    <div className="space-y-1">
                      {g.rows.map((r, idx) => (
                        <div key={`${g.year}-${idx}`} className="flex justify-between text-sm">
                          <span className="text-gray-800">{r.gender} {r.distance} {r.stroke} — {r.round} {r.place ? `(${r.place})` : ''} {r.school_name ? `- ${r.school_name}` : ''}</span>
                          <span className="font-semibold">{r.time_raw || formatTime(r.time_seconds)}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default Historical; 