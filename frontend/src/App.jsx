/**
 * @typedef {Object} PredictionItem
 * @property {string} ticker - Stock ticker symbol
 * @property {string} date - Date in ISO format (YYYY-MM-DD)
 * @property {string} headline - News headline text
 * @property {number} prob_up - Model probability that stock price will go up (0.0 to 1.0)
 * @property {string} prediction - "UP" or "DOWN" based on 0.5 threshold
 * @property {number|null} true_label - Actual label (0 or 1) if available, null otherwise
 */

/**
 * @typedef {Object} OverallSummary
 * @property {Object} all_predictions - Metrics for all predictions
 * @property {Object} with_headlines - Metrics for predictions with headlines only
 */

/**
 * @typedef {Object} TickerSummaryItem
 * @property {string} ticker - Ticker symbol
 * @property {number} n_total - Total number of predictions
 * @property {number} n_with_headline - Number of predictions with headlines
 * @property {number|null} accuracy_total - Accuracy for all predictions (percentage)
 * @property {number|null} accuracy_with_headline - Accuracy for predictions with headlines (percentage)
 * @property {number} avg_prob_up - Average probability
 * @property {number|null} avg_prob_up_with_headline - Average probability for predictions with headlines
 */

import Button from '@mui/material/Button';
import TextField from '@mui/material/TextField';
import GitHubIcon from '@mui/icons-material/GitHub';
import CircularProgress from '@mui/material/CircularProgress';
import Alert from '@mui/material/Alert';
import Table from '@mui/material/Table';
import TableBody from '@mui/material/TableBody';
import TableCell from '@mui/material/TableCell';
import TableContainer from '@mui/material/TableContainer';
import TableHead from '@mui/material/TableHead';
import TableRow from '@mui/material/TableRow';
import Paper from '@mui/material/Paper';
import Chip from '@mui/material/Chip';
import Box from '@mui/material/Box';
import FormControl from '@mui/material/FormControl';
import Select from '@mui/material/Select';
import MenuItem from '@mui/material/MenuItem';
import InputLabel from '@mui/material/InputLabel';
import Tooltip from '@mui/material/Tooltip';
import InfoOutlinedIcon from '@mui/icons-material/InfoOutlined';
import Switch from '@mui/material/Switch';
import FormControlLabel from '@mui/material/FormControlLabel';
import SearchIcon from '@mui/icons-material/Search';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';

import { useState, useMemo, useEffect } from 'react';

// API base URL from environment variable, fallback to default
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? import.meta.env.VITE_API_URL ?? 'http://localhost:8000';

/**
 * Fetches overall summary statistics
 */
async function fetchOverallSummary() {
  const url = `${API_BASE_URL}/summary/overall`;
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`API error: ${response.status} ${response.statusText}`);
  }
  return await response.json();
}

/**
 * Fetches ticker summary statistics
 */
async function fetchTickerSummary() {
  const url = `${API_BASE_URL}/summary/tickers`;
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`API error: ${response.status} ${response.statusText}`);
  }
  return await response.json();
}

/**
 * Fetches all predictions for a specific ticker
 * @param {string} ticker - Ticker symbol
 * @param {AbortSignal} [signal] - Optional abort signal for request cancellation
 */
async function fetchTickerPredictions(ticker, signal) {
  const url = `${API_BASE_URL}/ticker/${ticker}/predictions`;
  const response = await fetch(url, { signal });
  if (!response.ok) {
    throw new Error(`API error: ${response.status} ${response.statusText}`);
  }
  return await response.json();
}

/**
 * Searches predictions by headline text
 */
async function searchHeadlines(query) {
  const url = `${API_BASE_URL}/search/headlines?q=${encodeURIComponent(query)}`;
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`API error: ${response.status} ${response.statusText}`);
  }
  return await response.json();
}

/**
 * LivePredictionSection - Predict tomorrow using live news
 */
function LivePredictionSection() {
  const [ticker, setTicker] = useState('AAPL');
  const [headline, setHeadline] = useState('Apple reports record earnings for Q4.');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [result, setResult] = useState(null);

  const [tickerTomorrow, setTickerTomorrow] = useState('AAPL');
  const [loadingTomorrow, setLoadingTomorrow] = useState(false);
  const [errorTomorrow, setErrorTomorrow] = useState('');
  const [resultTomorrow, setResultTomorrow] = useState(null);

  const canSubmit = useMemo(() => {
    return Boolean(ticker.trim().length && headline.trim().length >= 5);
  }, [ticker, headline]);

  const canSubmitTomorrow = useMemo(() => {
    return Boolean(tickerTomorrow.trim().length);
  }, [tickerTomorrow]);

  const handlePredict = async () => {
    setLoading(true);
    setError('');
    setResult(null);
    try {
      const response = await fetch(`${API_BASE_URL}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ticker: ticker.trim(),
          headline: headline.trim(),
        }),
      });

      if (!response.ok) {
        const message = await response.text();
        throw new Error(message || 'Prediction failed');
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(err.message || 'Unexpected error occurred');
    } finally {
      setLoading(false);
    }
  };

  const handlePredictTomorrow = async () => {
    setLoadingTomorrow(true);
    setErrorTomorrow('');
    setResultTomorrow(null);
    try {
      const response = await fetch(`${API_BASE_URL}/predict-tomorrow`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ticker: tickerTomorrow.trim(),
        }),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: 'Prediction failed' }));
        throw new Error(errorData.detail || 'Prediction failed');
      }

      const data = await response.json();
      setResultTomorrow(data);
    } catch (err) {
      setErrorTomorrow(err.message || 'Unexpected error occurred');
    } finally {
      setLoadingTomorrow(false);
    }
  };

  return (
    <div className="bg-white shadow-sm rounded-xl p-6 md:p-8 border border-slate-100 w-full">
      <div className="mb-6">
        <div className="flex items-center gap-2 mb-2">
          <TrendingUpIcon className="text-emerald-600" />
          <h2 className="text-xl font-semibold text-slate-800">Live Predictions</h2>
        </div>
        <p className="text-sm text-slate-600">
          Make real-time predictions using current market data and news
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Manual Headline Prediction */}
        <div className="border border-slate-200 rounded-lg p-4 space-y-4">
          <h3 className="text-lg font-medium text-slate-800">Custom Headline</h3>
          <p className="text-sm text-slate-600">Enter your own headline to see prediction</p>

          <TextField
            label="Ticker symbol"
            value={ticker}
            onChange={(e) => setTicker(e.target.value.toUpperCase())}
            inputProps={{ maxLength: 8 }}
            fullWidth
            size="small"
          />

          <TextField
            label="News headline"
            value={headline}
            onChange={(e) => setHeadline(e.target.value)}
            multiline
            minRows={2}
            fullWidth
            size="small"
          />

          <Button
            variant="contained"
            disabled={!canSubmit || loading}
            sx={{ textTransform: 'none', borderRadius: '8px' }}
            onClick={handlePredict}
            fullWidth
          >
            {loading ? <CircularProgress size={24} color="inherit" /> : 'Predict Movement'}
          </Button>

          {error && <Alert severity="error" sx={{ mt: 2 }}>{error}</Alert>}

          {result && (
            <Alert severity={result.direction === 'UP' ? 'success' : 'warning'} sx={{ mt: 2 }}>
              {result.direction === 'UP'
                ? `${result.ticker} predicted UP (${(result.probability * 100).toFixed(2)}%)`
                : `${result.ticker} predicted DOWN (${(result.probability * 100).toFixed(2)}%)`}
            </Alert>
          )}
        </div>

        {/* Live News Prediction */}
        <div className="border border-slate-200 rounded-lg p-4 space-y-4">
          <h3 className="text-lg font-medium text-slate-800">Tomorrow's Prediction</h3>
          <p className="text-sm text-slate-600">Uses live news from Finnhub API</p>

          <TextField
            label="Ticker symbol"
            value={tickerTomorrow}
            onChange={(e) => setTickerTomorrow(e.target.value.toUpperCase())}
            inputProps={{ maxLength: 8 }}
            fullWidth
            size="small"
          />

          <Button
            variant="contained"
            disabled={!canSubmitTomorrow || loadingTomorrow}
            sx={{ textTransform: 'none', borderRadius: '8px' }}
            onClick={handlePredictTomorrow}
            fullWidth
          >
            {loadingTomorrow ? <CircularProgress size={24} color="inherit" /> : 'Predict Tomorrow'}
          </Button>

          {errorTomorrow && <Alert severity="error" sx={{ mt: 2 }}>{errorTomorrow}</Alert>}

          {resultTomorrow && (
            <Alert severity={resultTomorrow.direction === 'UP' ? 'success' : 'warning'} sx={{ mt: 2 }}>
              <div className="space-y-1">
                <div className="font-medium">Prediction for {resultTomorrow.date}:</div>
                <div>
                  {resultTomorrow.direction === 'UP'
                    ? `${resultTomorrow.ticker} predicted UP (${(resultTomorrow.probability * 100).toFixed(2)}%)`
                    : `${resultTomorrow.ticker} predicted DOWN (${(resultTomorrow.probability * 100).toFixed(2)}%)`}
                </div>
                <div className="text-xs opacity-75 mt-2">
                  Based on {resultTomorrow.headlines_used} headline{resultTomorrow.headlines_used !== 1 ? 's' : ''} from today
                </div>
              </div>
            </Alert>
          )}
        </div>
      </div>
    </div>
  );
}

/**
 * OverallSummaryDashboard - Shows aggregate metrics for all predictions vs with headlines
 */
function OverallSummaryDashboard({ overallSummary, isLoading }) {
  if (isLoading) {
    return (
      <div className="bg-white shadow-sm rounded-xl p-6 md:p-8 border border-slate-100 w-full">
        <div className="flex justify-center items-center py-8">
          <CircularProgress />
        </div>
      </div>
    );
  }

  if (!overallSummary) {
    return null;
  }

  const all = overallSummary.all_predictions;
  const withHeadlines = overallSummary.with_headlines;

  return (
    <div className="bg-white shadow-sm rounded-xl p-6 md:p-8 border border-slate-100 w-full">
      <div className="mb-4">
        <h2 className="text-xl font-semibold text-slate-800">Historical Model Performance</h2>
        <p className="text-sm text-slate-600 mt-1">
          Aggregate performance across the entire dataset (2018-2023)
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* All Predictions */}
        <div className="border border-slate-200 rounded-lg p-4">
          <h3 className="text-lg font-medium text-slate-800 mb-3">All Predictions</h3>
          <div className="space-y-2">
            <div className="flex justify-between">
              <span className="text-slate-600">Total Predictions:</span>
              <span className="font-semibold">{all.total.toLocaleString()}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-600">Bullish (UP):</span>
              <span className="font-semibold text-emerald-600">{all.up_count.toLocaleString()}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-600">Bearish (DOWN):</span>
              <span className="font-semibold text-rose-600">{all.down_count.toLocaleString()}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-600">Avg Confidence:</span>
              <span className="font-semibold">{(all.avg_prob_up * 100).toFixed(1)}%</span>
            </div>
            {all.accuracy !== null && (
              <div className="flex justify-between pt-2 border-t border-slate-200">
                <span className="text-slate-600">Accuracy:</span>
                <span className={`font-semibold ${all.accuracy >= 60 ? 'text-emerald-600' : all.accuracy >= 50 ? 'text-amber-600' : 'text-rose-600'}`}>
                  {all.accuracy.toFixed(1)}%
                </span>
              </div>
            )}
          </div>
        </div>

        {/* With Headlines Only */}
        <div className="border border-slate-200 rounded-lg p-4">
          <h3 className="text-lg font-medium text-slate-800 mb-3">With Headlines Only</h3>
          <div className="space-y-2">
            <div className="flex justify-between">
              <span className="text-slate-600">Total Predictions:</span>
              <span className="font-semibold">{withHeadlines.total.toLocaleString()}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-600">Bullish (UP):</span>
              <span className="font-semibold text-emerald-600">{withHeadlines.up_count.toLocaleString()}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-600">Bearish (DOWN):</span>
              <span className="font-semibold text-rose-600">{withHeadlines.down_count.toLocaleString()}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-600">Avg Confidence:</span>
              <span className="font-semibold">{(withHeadlines.avg_prob_up * 100).toFixed(1)}%</span>
            </div>
            {withHeadlines.accuracy !== null && (
              <div className="flex justify-between pt-2 border-t border-slate-200">
                <span className="text-slate-600">Accuracy:</span>
                <span className={`font-semibold ${withHeadlines.accuracy >= 60 ? 'text-emerald-600' : withHeadlines.accuracy >= 50 ? 'text-amber-600' : 'text-rose-600'}`}>
                  {withHeadlines.accuracy.toFixed(1)}%
                </span>
              </div>
            )}
          </div>
        </div>
      </div>

      <div className="mt-4 text-xs text-slate-500">
        <InfoOutlinedIcon sx={{ fontSize: 16, verticalAlign: 'middle', mr: 0.5 }} />
        Predictions with headlines are those where news articles were successfully mapped to the stock ticker.
      </div>
    </div>
  );
}

/**
 * TickerSummaryTable - Table showing performance for each ticker
 */
function TickerSummaryTable({ tickerSummary, isLoading, tickerFilter, onTickerClick, onTickerFilterChange }) {
  const filteredTickers = useMemo(() => {
    if (!tickerSummary || !tickerSummary.tickers) return [];

    let filtered = tickerSummary.tickers;

    if (tickerFilter) {
      const filterLower = tickerFilter.toLowerCase();
      filtered = filtered.filter(t => t.ticker.toLowerCase().includes(filterLower));
    }

    return filtered;
  }, [tickerSummary, tickerFilter]);

  if (isLoading) {
    return (
      <div className="bg-white shadow-sm rounded-xl p-6 md:p-8 border border-slate-100 w-full">
        <div className="flex justify-center items-center py-8">
          <CircularProgress />
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white shadow-sm rounded-xl p-6 md:p-8 border border-slate-100 w-full">
      <div className="mb-4 flex items-center justify-between flex-wrap gap-4">
        <h3 className="text-lg font-semibold text-slate-800">Ticker Performance Summary</h3>
        <div className="flex items-center gap-4">
          <TextField
            size="small"
            label="Filter by ticker"
            value={tickerFilter}
            onChange={(e) => onTickerFilterChange(e.target.value)}
            sx={{ minWidth: 200 }}
          />
        </div>
      </div>

      <TableContainer component={Paper} sx={{ maxHeight: 500 }} className="border border-slate-100">
        <Table stickyHeader size="small">
          <TableHead>
            <TableRow>
              <TableCell sx={{ fontWeight: 600 }}>Ticker</TableCell>
              <TableCell sx={{ fontWeight: 600 }} align="right"># Predictions (All)</TableCell>
              <TableCell sx={{ fontWeight: 600 }} align="right"># With Headlines</TableCell>
              <TableCell sx={{ fontWeight: 600 }} align="right">Accuracy (All)</TableCell>
              <TableCell sx={{ fontWeight: 600 }} align="right">Accuracy (With Headlines)</TableCell>
              <TableCell sx={{ fontWeight: 600 }} align="right">Avg Confidence</TableCell>
              <TableCell sx={{ fontWeight: 600 }}>Actions</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {filteredTickers.length === 0 ? (
              <TableRow>
                <TableCell colSpan={7} align="center" className="py-8 text-slate-500">
                  No tickers found matching the filters.
                </TableCell>
              </TableRow>
            ) : (
              filteredTickers.map((ticker) => (
                <TableRow key={ticker.ticker} hover>
                  <TableCell sx={{ fontWeight: 600 }}>{ticker.ticker}</TableCell>
                  <TableCell align="right">{ticker.n_total.toLocaleString()}</TableCell>
                  <TableCell align="right">{ticker.n_with_headline.toLocaleString()}</TableCell>
                  <TableCell align="right">
                    {ticker.accuracy_total !== null ? `${ticker.accuracy_total.toFixed(1)}%` : 'N/A'}
                  </TableCell>
                  <TableCell align="right">
                    {ticker.accuracy_with_headline !== null ? `${ticker.accuracy_with_headline.toFixed(1)}%` : 'N/A'}
                  </TableCell>
                  <TableCell align="right">{(ticker.avg_prob_up * 100).toFixed(1)}%</TableCell>
                  <TableCell>
                    <Button
                      size="small"
                      variant="outlined"
                      onClick={() => onTickerClick(ticker.ticker)}
                    >
                      View Details
                    </Button>
                  </TableCell>
                </TableRow>
              ))
            )}
          </TableBody>
        </Table>
      </TableContainer>
    </div>
  );
}

/**
 * TickerDetailPanel - Shows detailed predictions for a selected ticker
 */
function TickerDetailPanel({ ticker, tickerDetail, isLoading, showWithHeadlinesOnly, onShowWithHeadlinesChange, onClose }) {
  const filteredItems = useMemo(() => {
    if (!tickerDetail || !tickerDetail.items) return [];

    let items = tickerDetail.items;

    if (showWithHeadlinesOnly) {
      items = items.filter(item =>
        item.headline && item.headline.trim() && item.headline.toLowerCase() !== 'nan'
      );
    }

    return items;
  }, [tickerDetail, showWithHeadlinesOnly]);

  if (!ticker) {
    return null;
  }

  return (
    <div className="bg-white shadow-sm rounded-xl p-6 md:p-8 border border-slate-100 w-full">
      <div className="mb-4 flex items-center justify-between">
        <div>
          <h3 className="text-lg font-semibold text-slate-800">Ticker: {ticker}</h3>
          <p className="text-sm text-slate-600 mt-1">
            {tickerDetail ? `${tickerDetail.count} total predictions` : 'Loading...'}
          </p>
        </div>
        <div className="flex items-center gap-4">
          <FormControlLabel
            control={
              <Switch
                checked={showWithHeadlinesOnly}
                onChange={(e) => onShowWithHeadlinesChange(e.target.checked)}
              />
            }
            label="With headlines only"
          />
          <Button variant="outlined" onClick={onClose}>Close</Button>
        </div>
      </div>

      {isLoading ? (
        <div className="flex justify-center items-center py-8">
          <CircularProgress />
        </div>
      ) : filteredItems.length === 0 ? (
        <div className="text-center py-8 text-slate-500">
          <InfoOutlinedIcon sx={{ fontSize: 48, color: 'rgb(148 163 184)', mb: 2 }} />
          <p>No predictions found for this ticker.</p>
        </div>
      ) : (
        <TableContainer component={Paper} sx={{ maxHeight: 500 }} className="border border-slate-100">
          <Table stickyHeader size="small">
            <TableHead>
              <TableRow>
                <TableCell sx={{ fontWeight: 600 }}>Date</TableCell>
                <TableCell sx={{ fontWeight: 600 }}>Headline</TableCell>
                <TableCell sx={{ fontWeight: 600 }}>Confidence</TableCell>
                <TableCell sx={{ fontWeight: 600 }}>Prediction</TableCell>
                <TableCell sx={{ fontWeight: 600 }}>Actual vs Predicted</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {filteredItems.map((item) => {
                const isCorrect = item.true_label !== null
                  ? ((item.prob_up > 0.5) === (item.true_label === 1))
                  : null;
                const trueLabelText = item.true_label !== null
                  ? (item.true_label === 1 ? 'UP' : 'DOWN')
                  : null;

                const stableKey = `${item.ticker}-${item.date}-${(item.headline || '').substring(0, 30)}`;

                return (
                  <TableRow key={stableKey} hover>
                    <TableCell>{item.date}</TableCell>
                    <TableCell>
                      <Tooltip title={item.headline || 'No headline available'} arrow>
                        <span className="text-sm text-slate-700 cursor-help">
                          {(item.headline || 'No headline available').length > 60
                            ? `${(item.headline || 'No headline available').substring(0, 60)}...`
                            : (item.headline || 'No headline available')}
                        </span>
                      </Tooltip>
                    </TableCell>
                    <TableCell>
                      <div className="flex flex-col gap-1">
                        <span
                          style={{
                            color: `rgb(${Math.round(255 * (1 - item.prob_up))}, ${Math.round(255 * item.prob_up)}, 0)`,
                            fontWeight: 'bold',
                            fontSize: '0.875rem'
                          }}
                        >
                          {(item.prob_up * 100).toFixed(2)}%
                        </span>
                        <div className="w-full h-1.5 rounded-full bg-slate-200 overflow-hidden">
                          <div
                            className={`h-full rounded-full ${item.prob_up > 0.5 ? 'bg-emerald-500' : 'bg-rose-500'}`}
                            style={{ width: `${item.prob_up * 100}%` }}
                          />
                        </div>
                      </div>
                    </TableCell>
                    <TableCell>
                      <Chip
                        label={item.prediction}
                        color={item.prediction === 'UP' ? 'success' : 'error'}
                        size="small"
                      />
                    </TableCell>
                    <TableCell>
                      {item.true_label !== null ? (
                        <div className="flex items-center gap-2">
                          <Chip
                            label={isCorrect ? 'Correct' : 'Incorrect'}
                            color={isCorrect ? 'success' : 'error'}
                            size="small"
                            variant="outlined"
                          />
                          <span className="text-xs text-slate-500">(True: {trueLabelText})</span>
                        </div>
                      ) : (
                        <span className="text-slate-400 text-sm">N/A</span>
                      )}
                    </TableCell>
                  </TableRow>
                );
              })}
            </TableBody>
          </Table>
        </TableContainer>
      )}
    </div>
  );
}

/**
 * HeadlineSearchSection - Search predictions by headline text
 */
function HeadlineSearchSection({ searchQuery, searchResults, isLoading, onSearchChange, onSearch, onTickerClick }) {
  return (
    <div className="bg-white shadow-sm rounded-xl p-6 md:p-8 border border-slate-100 w-full">
      <div className="mb-4">
        <h3 className="text-lg font-semibold text-slate-800">Search by Headline</h3>
        <p className="text-sm text-slate-600 mt-1">
          Search historical predictions by headline text (case-insensitive)
        </p>
      </div>

      <Box sx={{ display: 'flex', gap: 2, mb: 4 }}>
        <TextField
          fullWidth
          label="Search headlines"
          value={searchQuery}
          onChange={(e) => onSearchChange(e.target.value)}
          onKeyPress={(e) => {
            if (e.key === 'Enter') {
              onSearch();
            }
          }}
          InputProps={{
            startAdornment: <SearchIcon sx={{ mr: 1, color: 'text.secondary' }} />
          }}
        />
        <Button
          variant="contained"
          onClick={onSearch}
          disabled={isLoading || !searchQuery.trim()}
          sx={{ textTransform: "none", borderRadius: "8px" }}
        >
          {isLoading ? <CircularProgress size={20} /> : 'Search'}
        </Button>
      </Box>

      {searchResults && searchResults.count > 0 && (
        <div>
          <p className="text-sm text-slate-600 mb-2">
            Found {searchResults.count} prediction{searchResults.count !== 1 ? 's' : ''} for "{searchQuery}"
          </p>
          <TableContainer component={Paper} sx={{ maxHeight: 400 }} className="border border-slate-100">
            <Table stickyHeader size="small">
              <TableHead>
                <TableRow>
                  <TableCell sx={{ fontWeight: 600 }}>Date</TableCell>
                  <TableCell sx={{ fontWeight: 600 }}>Ticker</TableCell>
                  <TableCell sx={{ fontWeight: 600 }}>Headline</TableCell>
                  <TableCell sx={{ fontWeight: 600 }}>Confidence</TableCell>
                  <TableCell sx={{ fontWeight: 600 }}>Prediction</TableCell>
                  <TableCell sx={{ fontWeight: 600 }}>Actions</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {searchResults.items.map((item) => {
                  const stableKey = `${item.ticker}-${item.date}-${(item.headline || '').substring(0, 30)}`;
                  return (
                    <TableRow key={stableKey} hover>
                      <TableCell>{item.date}</TableCell>
                      <TableCell sx={{ fontWeight: 600 }}>{item.ticker}</TableCell>
                      <TableCell>
                        <Tooltip title={item.headline || 'No headline available'} arrow>
                          <span className="text-sm text-slate-700 cursor-help">
                            {(item.headline || 'No headline available').length > 60
                              ? `${(item.headline || 'No headline available').substring(0, 60)}...`
                              : (item.headline || 'No headline available')}
                          </span>
                        </Tooltip>
                      </TableCell>
                      <TableCell>
                        <span
                          style={{
                            color: `rgb(${Math.round(255 * (1 - item.prob_up))}, ${Math.round(255 * item.prob_up)}, 0)`,
                            fontWeight: 'bold',
                            fontSize: '0.875rem'
                          }}
                        >
                          {(item.prob_up * 100).toFixed(2)}%
                        </span>
                      </TableCell>
                      <TableCell>
                        <Chip
                          label={item.prediction}
                          color={item.prediction === 'UP' ? 'success' : 'error'}
                          size="small"
                        />
                      </TableCell>
                      <TableCell>
                        <Button
                          size="small"
                          variant="outlined"
                          onClick={() => onTickerClick(item.ticker)}
                        >
                          View Ticker
                        </Button>
                      </TableCell>
                    </TableRow>
                  );
                })}
              </TableBody>
            </Table>
          </TableContainer>
        </div>
      )}

      {searchResults && searchResults.count === 0 && searchQuery && (
        <div className="text-center py-8 text-slate-500">
          <InfoOutlinedIcon sx={{ fontSize: 48, color: 'rgb(148 163 184)', mb: 2 }} />
          <p>No predictions found matching "{searchQuery}"</p>
        </div>
      )}
    </div>
  );
}

function App() {
  // State management
  const [overallSummary, setOverallSummary] = useState(null);
  const [tickerSummary, setTickerSummary] = useState(null);
  const [selectedTicker, setSelectedTicker] = useState(null);
  const [tickerDetail, setTickerDetail] = useState(null);
  const [headlineSearchQuery, setHeadlineSearchQuery] = useState('');
  const [headlineSearchResults, setHeadlineSearchResults] = useState(null);

  const [isLoadingOverall, setIsLoadingOverall] = useState(true);
  const [isLoadingTickers, setIsLoadingTickers] = useState(true);
  const [isLoadingTickerDetail, setIsLoadingTickerDetail] = useState(false);
  const [isLoadingSearch, setIsLoadingSearch] = useState(false);

  const [error, setError] = useState(null);

  const [tickerFilter, setTickerFilter] = useState('');
  const [tickerDetailShowWithHeadlinesOnly, setTickerDetailShowWithHeadlinesOnly] = useState(false);

  // Load overall summary and ticker summary on mount
  useEffect(() => {
    async function loadInitialData() {
      setIsLoadingOverall(true);
      setIsLoadingTickers(true);
      setError(null);

      try {
        const [overall, tickers] = await Promise.all([
          fetchOverallSummary(),
          fetchTickerSummary()
        ]);

        setOverallSummary(overall);
        setTickerSummary(tickers);
      } catch (err) {
        setError(err.message || 'Failed to load initial data. Is the backend running?');
      } finally {
        setIsLoadingOverall(false);
        setIsLoadingTickers(false);
      }
    }

    loadInitialData();
  }, []);

  // Load ticker detail when a ticker is selected
  useEffect(() => {
    if (!selectedTicker) {
      setTickerDetail(null);
      return;
    }

    const abortController = new AbortController();
    const signal = abortController.signal;

    async function loadTickerDetail() {
      setIsLoadingTickerDetail(true);
      setError(null);

      try {
        const data = await fetchTickerPredictions(selectedTicker, signal);
        if (!signal.aborted) {
          setTickerDetail(data);
        }
      } catch (err) {
        if (err.name === 'AbortError' || signal.aborted) {
          return;
        }
        setError(err.message || `Failed to load predictions for ${selectedTicker}`);
      } finally {
        if (!signal.aborted) {
          setIsLoadingTickerDetail(false);
        }
      }
    }

    loadTickerDetail();

    return () => {
      abortController.abort();
    };
  }, [selectedTicker]);

  // Clear search results when search query changes
  useEffect(() => {
    if (headlineSearchResults && headlineSearchQuery !== headlineSearchResults.query) {
      setHeadlineSearchResults(null);
    }
  }, [headlineSearchQuery]);

  const handleTickerClick = (ticker) => {
    setSelectedTicker(ticker);
    setTimeout(() => {
      const element = document.getElementById('ticker-detail-panel');
      if (element) {
        element.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }
    }, 100);
  };

  const handleSearch = async () => {
    if (!headlineSearchQuery.trim()) {
      return;
    }

    setIsLoadingSearch(true);
    setError(null);

    try {
      const results = await searchHeadlines(headlineSearchQuery);
      setHeadlineSearchResults(results);
    } catch (err) {
      setError(err.message || 'Failed to search headlines');
      setHeadlineSearchResults(null);
    } finally {
      setIsLoadingSearch(false);
    }
  };

  return (
    <div className="flex flex-col items-center min-h-screen py-12 px-6 bg-[#F3F6FB]">
      {/* Header */}
      <div className="w-full max-w-6xl mb-10">
        <div className="text-4xl font-semibold text-slate-800">ECS171: News Market Predictor</div>
        <Button
          variant="contained"
          startIcon={<GitHubIcon />}
          href="https://github.com/honoreaa/News-Market-Predictions"
          target="_blank"
          sx={{ textTransform: "none", borderRadius: "8px", mt: 1 }}
        >
          Project Github
        </Button>
        <div className="text-sm mt-1 leading-relaxed text-slate-600">
          Group 4: Honore Alexander, Owen Holt, Pranavi Khanna, Dylan Lim,
          Yihong Li, Ethan Lee, Dan Firstenberg, Hyeongseung Nam, Oscar Pineda,
          Kevin Zhang, Zachary Chan, Vicente Aguayo
        </div>
      </div>

      {/* Error Display */}
      {error && (
        <div className="w-full max-w-6xl mb-4">
          <Alert
            severity="error"
            onClose={() => setError(null)}
            sx={{ borderRadius: '8px' }}
          >
            {error}
          </Alert>
        </div>
      )}

      {/* Live Prediction Section */}
      <div className="w-full max-w-6xl mb-6">
        <LivePredictionSection />
      </div>

      {/* Overall Summary Dashboard */}
      <div className="w-full max-w-6xl mb-6">
        <OverallSummaryDashboard overallSummary={overallSummary} isLoading={isLoadingOverall} />
      </div>

      {/* Ticker Summary Table */}
      <div className="w-full max-w-6xl mb-6">
        <TickerSummaryTable
          tickerSummary={tickerSummary}
          isLoading={isLoadingTickers}
          tickerFilter={tickerFilter}
          onTickerClick={handleTickerClick}
          onTickerFilterChange={setTickerFilter}
        />
      </div>

      {/* Headline Search Section */}
      <div className="w-full max-w-6xl mb-6">
        <HeadlineSearchSection
          searchQuery={headlineSearchQuery}
          searchResults={headlineSearchResults}
          isLoading={isLoadingSearch}
          onSearchChange={setHeadlineSearchQuery}
          onSearch={handleSearch}
          onTickerClick={handleTickerClick}
        />
      </div>

      {/* Ticker Detail Panel */}
      {selectedTicker && (
        <div id="ticker-detail-panel" className="w-full max-w-6xl mb-6">
          <TickerDetailPanel
            ticker={selectedTicker}
            tickerDetail={tickerDetail}
            isLoading={isLoadingTickerDetail}
            showWithHeadlinesOnly={tickerDetailShowWithHeadlinesOnly}
            onShowWithHeadlinesChange={setTickerDetailShowWithHeadlinesOnly}
            onClose={() => setSelectedTicker(null)}
          />
        </div>
      )}
    </div>
  );
}

export default App;
