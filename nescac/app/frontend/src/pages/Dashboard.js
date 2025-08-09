import React, { useState, useEffect, useMemo } from 'react';
import { School, BarChart, ChevronLeft, ChevronRight } from 'lucide-react';
import axios from 'axios';

const EVENTS = [
  '50_Freestyle','100_Freestyle','200_Freestyle','500_Freestyle',
  '50_Backstroke','100_Backstroke','200_Backstroke',
  '50_Breaststroke','100_Breaststroke','200_Breaststroke',
  '50_Butterfly','100_Butterfly','200_Butterfly',
  '200_IM','400_IM'
];

// Minimal inline SVG line chart for time-series (retained only as fallback for school if needed)
const LineChart = ({ series, width = 800, height = 280, color = '#2563eb', yLabel = 'sec' }) => {
  if (!series || !series.years || !series.values || series.years.length === 0) {
    return <div className="text-sm text-gray-500 text-center">No data</div>;
  }
  const margin = { top: 10, right: 16, bottom: 24, left: 40 };
  const w = width - margin.left - margin.right;
  const h = height - margin.top - margin.bottom;
  const xs = series.years;
  const ys = series.values;
  const xMin = Math.min(...xs);
  const xMax = Math.max(...xs);
  const yMin = Math.min(...ys);
  const yMax = Math.max(...ys);
  const xScale = (x) => w * (x - xMin) / Math.max(1, (xMax - xMin));
  const yScale = (y) => h - (h * (y - yMin) / Math.max(1e-9, (yMax - yMin)));
  const points = xs.map((x, i) => [xScale(x), yScale(ys[i])]);
  const d = points.map((p, i) => `${i === 0 ? 'M' : 'L'} ${p[0].toFixed(2)} ${p[1].toFixed(2)}`).join(' ');

  const yTicks = 4;
  const tickVals = Array.from({ length: yTicks + 1 }, (_, i) => yMin + (i * (yMax - yMin)) / yTicks);

  return (
    <svg width={width} height={height} className="w-full">
      <g transform={`translate(${margin.left},${margin.top})`}>
        {/* Y grid + ticks */}
        {tickVals.map((t, i) => {
          const y = yScale(t);
          return (
            <g key={i}>
              <line x1={0} x2={w} y1={y} y2={y} stroke="#e5e7eb" />
              <text x={-8} y={y} textAnchor="end" alignmentBaseline="middle" fontSize="11" fill="#6b7280">
                {formatTime(t)}
              </text>
            </g>
          );
        })}
        {/* X axis min/max labels */}
        <text x={0} y={h + 16} fontSize="11" fill="#6b7280">{xMin}</text>
        <text x={w} y={h + 16} fontSize="11" textAnchor="end" fill="#6b7280">{xMax}</text>
        {/* Path */}
        <path d={d} fill="none" stroke={color} strokeWidth={2} />
      </g>
    </svg>
  );
};

// Format seconds into MM:SS.00
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

const Dashboard = () => {
  // Core selections
  const [schools, setSchools] = useState([]);
  const [selectedSchool, setSelectedSchool] = useState('');
  const [selectedEvent, setSelectedEvent] = useState('100_Freestyle');

  // Views
  const [viewMode, setViewMode] = useState('event'); // 'event' | 'school'

  // Data
  const [eventStats, setEventStats] = useState(null); // selected event details

  // Gallery index
  const [galleryIndex, setGalleryIndex] = useState(0);

  // Track if a winning image failed to load for a given event
  const [winningImgError, setWinningImgError] = useState({}); // ev -> true if error
  // Track if a school individual-event image failed to load for a given (school|event)
  const [schoolImgError, setSchoolImgError] = useState({}); // key `${school}|${event}` -> true

  // School accordion data: map `${school}|${event}` -> stats
  const [schoolEventStatsMap, setSchoolEventStatsMap] = useState({});
  const [expandedEvents, setExpandedEvents] = useState({}); // event -> boolean

  const getWinningImagePath = (ev) => `/api/plots/winning_times/Men_${ev}_winning.png`;

  // Boot: fetch schools
  useEffect(() => {
    const boot = async () => {
      try {
        const schoolsRes = await axios.get('/api/schools');
        const fetchedSchools = schoolsRes.data || [];
        setSchools(fetchedSchools);
        if (fetchedSchools.length > 0) {
          setSelectedSchool(fetchedSchools[0]);
        }
      } catch (e) {
        // eslint-disable-next-line no-console
        console.error(e);
      }
    };
    boot();
  }, []);

  // Load selected event stats for numeric summary
  useEffect(() => {
    const fetchEventStats = async () => {
      try {
        const [distance, ...strokeParts] = selectedEvent.split('_');
        const stroke = strokeParts.join('_');
        const res = await axios.get('/api/stats/event', {
          params: { gender: 'Men', stroke, distance: Number(distance) }
        });
        setEventStats(res.data);
      } catch (err) {
        // eslint-disable-next-line no-console
        console.error(err);
      }
    };
    fetchEventStats();
  }, [selectedEvent]);

  // School accordion: fetch per-event stats for the selected school lazily when expanded
  const fetchSchoolEventStats = async (school, ev) => {
    const key = `${school}|${ev}`;
    if (schoolEventStatsMap[key]) return; // cached
    try {
      const res = await axios.get('/api/stats/school', {
        params: { school, event_name: ev }
      });
      setSchoolEventStatsMap((prev) => ({ ...prev, [key]: res.data }));
    } catch (err) {
      // eslint-disable-next-line no-console
      console.error(err);
    }
  };

  const toggleExpandEvent = async (ev) => {
    setExpandedEvents((prev) => {
      const next = { ...prev, [ev]: !prev[ev] };
      return next;
    });
    if (!expandedEvents[ev] && selectedSchool) {
      await fetchSchoolEventStats(selectedSchool, ev);
    }
  };

  // Gallery helpers
  const galleryEvent = useMemo(() => EVENTS[galleryIndex % EVENTS.length], [galleryIndex]);

  const nextSlide = () => setGalleryIndex((i) => (i + 1) % EVENTS.length);
  const prevSlide = () => setGalleryIndex((i) => (i - 1 + EVENTS.length) % EVENTS.length);

  const eventLabel = (ev) => `Men ${ev.replace('_', ' ')}`;

  const eventCutoffImage = `/api/plots/event_cutoffs/Men_${selectedEvent}_cutoffs.png`;

  return (
    <div className="space-y-6">
      <div className="bg-white rounded-lg shadow p-6">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">NESCAC Swimming Dashboard</h1>
      </div>

      {/* Gallery: per-event winning time charts (prefer pre-generated images) */}
      <div className="bg-white rounded-lg shadow p-8">
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-xl font-semibold text-gray-900">Winning Times Gallery</h2>
          <div className="flex items-center space-x-2">
            <button onClick={prevSlide} className="p-2 rounded bg-gray-100 hover:bg-gray-200" aria-label="Previous">
              <ChevronLeft className="h-5 w-5 text-gray-700" />
            </button>
            <button onClick={nextSlide} className="p-2 rounded bg-gray-100 hover:bg-gray-200" aria-label="Next">
              <ChevronRight className="h-5 w-5 text-gray-700" />
            </button>
          </div>
        </div>
        <div className="cursor-pointer flex justify-center" onClick={() => setSelectedEvent(galleryEvent)}>
          <div className="w-full max-w-6xl px-2 md:px-4">
            <img
              src={getWinningImagePath(galleryEvent)}
              alt={`${eventLabel(galleryEvent)} winning times`}
              className="w-full h-auto object-contain rounded"
              onError={() => setWinningImgError((prev) => ({ ...prev, [galleryEvent]: true }))}
            />
          </div>
        </div>
        <div className="text-xs text-gray-500 mt-3 text-center">Click the chart to focus this event below</div>
      </div>

      {/* Mode selector */}
      <div className="bg-white rounded-lg shadow p-6">
        <div className="flex items-center space-x-6 mb-4">
          <label className="inline-flex items-center space-x-2">
            <input
              type="radio"
              name="viewMode"
              className="h-4 w-4"
              value="event"
              checked={viewMode === 'event'}
              onChange={() => setViewMode('event')}
            />
            <span className="text-gray-800 font-medium">By Event</span>
          </label>
          <label className="inline-flex items-center space-x-2">
            <input
              type="radio"
              name="viewMode"
              className="h-4 w-4"
              value="school"
              checked={viewMode === 'school'}
              onChange={() => setViewMode('school')}
            />
            <span className="text-gray-800 font-medium">By School</span>
          </label>
        </div>

        {viewMode === 'event' && (
          <div className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Event</label>
                <select className="w-full border rounded-md p-2" value={selectedEvent} onChange={(e) => setSelectedEvent(e.target.value)}>
                  {EVENTS.map((ev) => (
                    <option key={ev} value={ev}>{eventLabel(ev)}</option>
                  ))}
                </select>
              </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <div className="bg-white rounded-lg border p-6">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-md font-semibold text-gray-900">Winning time by year</h3>
                  <BarChart className="h-5 w-5 text-gray-500" />
                </div>
                <div className="flex justify-center">
                  <div className="w-full max-w-6xl px-2 md:px-4">
                    <img
                      src={getWinningImagePath(selectedEvent)}
                      alt={`${eventLabel(selectedEvent)} winning times`}
                      className="w-full h-auto object-contain rounded"
                      onError={() => setWinningImgError((prev) => ({ ...prev, [selectedEvent]: true }))}
                    />
                  </div>
                </div>
              </div>
              <div className="bg-white rounded-lg border p-6">
                <h3 className="text-md font-semibold text-gray-900 mb-4 text-center">Event cutoff by year</h3>
                <div className="flex justify-center">
                  <img src={eventCutoffImage} alt="Event cutoff analysis" className="w-full max-w-6xl rounded object-contain" />
                </div>
              </div>
            </div>

            {eventStats?.stats && (
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="bg-white rounded-lg shadow p-4 text-center">
                  <p className="text-sm text-gray-500">Win Start</p>
                  <p className="text-2xl font-semibold text-gray-900">{formatTime(eventStats.stats.winning_time_start ?? 0)}</p>
                  <p className="text-xs text-gray-500 mt-1">Start</p>
                </div>
                <div className="bg-white rounded-lg shadow p-4 text-center">
                  <p className="text-sm text-gray-500">Win End</p>
                  <p className="text-2xl font-semibold text-gray-900">{formatTime(eventStats.stats.winning_time_end ?? 0)}</p>
                  <p className="text-xs text-gray-500 mt-1">Most recent</p>
                </div>
                <div className="bg-white rounded-lg shadow p-4 text-center">
                  <p className="text-sm text-gray-500">Avg/Year (win)</p>
                  <p className="text-2xl font-semibold text-gray-900">{(eventStats.stats.winning_time_avg_annual_change ?? 0).toFixed(3)}s</p>
                  <p className="text-xs text-gray-500 mt-1">Average annual change</p>
                </div>
                {eventStats.stats.a_final_cutoff_start !== undefined && (
                  <>
                    <div className="bg-white rounded-lg shadow p-4 text-center">
                      <p className="text-sm text-gray-500">A Cut Start</p>
                      <p className="text-2xl font-semibold text-gray-900">{formatTime(eventStats.stats.a_final_cutoff_start)}</p>
                    </div>
                    <div className="bg-white rounded-lg shadow p-4 text-center">
                      <p className="text-sm text-gray-500">A Cut End</p>
                      <p className="text-2xl font-semibold text-gray-900">{formatTime(eventStats.stats.a_final_cutoff_end)}</p>
                    </div>
                  </>
                )}
              </div>
            )}
          </div>
        )}

        {viewMode === 'school' && (
          <div className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">School</label>
                <div className="flex items-center space-x-2">
                  <School className="h-5 w-5 text-gray-500" />
                  <select className="flex-1 border rounded-md p-2" value={selectedSchool} onChange={(e) => setSelectedSchool(e.target.value)}>
                    {schools.map((s) => (
                      <option key={s} value={s}>{s.replace('_', ' ')}</option>
                    ))}
                  </select>
                </div>
              </div>
            </div>

            <div className="divide-y rounded border bg-white">
              {EVENTS.map((ev) => {
                const isOpen = !!expandedEvents[ev];
                const key = `${selectedSchool}|${ev}`;
                const sData = schoolEventStatsMap[key];
                return (
                  <div key={ev}>
                    <button
                      className="w-full flex items-center justify-between px-4 py-3 hover:bg-gray-50"
                      onClick={() => toggleExpandEvent(ev)}
                    >
                      <span className="text-left font-medium text-gray-900">{ev.replace('_', ' ')}</span>
                      <span className={`transform transition ${isOpen ? 'rotate-90' : ''}`}>
                        <ChevronRight className="h-4 w-4 text-gray-600" />
                      </span>
                    </button>
                    {isOpen && (
                      <div className="px-4 pb-4 space-y-4">
                        <div className="grid grid-cols-1">
                          <div className="bg-white rounded-lg border p-8">
                            <div className="flex justify-center">
                              <div className="w-full max-w-screen-2xl px-2 md:px-8">
                                {schoolImgError[key] ? (
                                  sData && sData.years && sData.fastest_time_sec ? (
                                    <LineChart series={{ years: sData.years, values: sData.fastest_time_sec }} />
                                  ) : (
                                    <div className="text-sm text-gray-500 text-center">Loading...</div>
                                  )
                                ) : (
                                  <img
                                    src={`/api/plots/schools/individual-event/${encodeURIComponent(selectedSchool)}/${ev}.png`}
                                    alt={`${selectedSchool.replace('_',' ')} ${ev.replace('_',' ')} fastest by year`}
                                    className="w-full h-auto object-contain rounded"
                                    style={{ maxHeight: '70vh' }}
                                    onError={() => setSchoolImgError((prev) => ({ ...prev, [key]: true }))}
                                  />
                                )}
                              </div>
                            </div>
                          </div>
                        </div>

                        {sData && (
                          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                            {sData.start_time !== undefined && (
                              <div className="bg-white rounded-lg shadow p-4 text-center">
                                <p className="text-sm text-gray-500">School start</p>
                                <p className="text-2xl font-semibold text-gray-900">{formatTime(sData.start_time)}</p>
                              </div>
                            )}
                            {sData.end_time !== undefined && (
                              <div className="bg-white rounded-lg shadow p-4 text-center">
                                <p className="text-sm text-gray-500">School end</p>
                                <p className="text-2xl font-semibold text-gray-900">{formatTime(sData.end_time)}</p>
                              </div>
                            )}
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default Dashboard; 