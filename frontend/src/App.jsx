import { useMemo, useState } from 'react';
import Button from '@mui/material/Button';
import CircularProgress from '@mui/material/CircularProgress';
import GitHubIcon from '@mui/icons-material/GitHub';
import TextField from '@mui/material/TextField';
import Alert from '@mui/material/Alert';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

function App() {
  const [ticker, setTicker] = useState('AAPL');
  const [headline, setHeadline] = useState('Apple reports record earnings for Q4.');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [result, setResult] = useState(null);
  
  // Predict Tomorrow state
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
    <div className="flex flex-col items-center min-h-screen py-12 px-6 bg-[#F3F6FB]">
      <div className="w-full max-w-3xl mb-10 space-y-3">
        <div className="text-4xl font-semibold">ECS171: News Market Predictor</div>
        <Button
          variant="contained"
          startIcon={<GitHubIcon />}
          href="https://github.com/honoreaa/News-Market-Predictions"
          target="_blank"
          sx={{ textTransform: 'none', borderRadius: '8px' }}
        >
          Project Github
        </Button>
        <div className="text-sm mt-1 leading-relaxed opacity-80">
          Group 4: Honore Alexander, Owen Holt, Pranavi Khanna, Dylan Lim,
          Yihong Li, Ethan Lee, Dan Firstenberg, Hyeongseung Nam, Oscar Pineda,
          Kevin Zhang, Zachary Chan, Vicente Aguayo
        </div>
      </div>

      <div className="bg-white shadow-md rounded-xl p-8 w-full max-w-2xl space-y-6">
        <div className="text-lg font-medium">Prediction Inputs</div>

        <TextField
          label="Ticker symbol"
          value={ticker}
          onChange={(e) => setTicker(e.target.value.toUpperCase())}
          inputProps={{ maxLength: 8 }}
          fullWidth
        />

        <TextField
          label="News headline"
          value={headline}
          onChange={(e) => setHeadline(e.target.value)}
          multiline
          minRows={2}
          fullWidth
        />

        <div className="flex items-center gap-4">
          <Button
            variant="contained"
            disabled={!canSubmit || loading}
            sx={{ textTransform: 'none', borderRadius: '8px' }}
            onClick={handlePredict}
          >
            {loading ? <CircularProgress size={24} color="inherit" /> : 'Predict Movement'}
          </Button>
          <div className="text-sm opacity-70">
            API base: {API_BASE_URL}
          </div>
        </div>

        {error && <Alert severity="error">{error}</Alert>}

        {result && (
          <Alert severity={result.direction === 'UP' ? 'success' : 'warning'}>
            {result.direction === 'UP'
              ? `The model predicts ${result.ticker} will go UP (${(result.probability * 100).toFixed(2)}% confidence).`
              : `The model predicts ${result.ticker} will go DOWN (${(result.probability * 100).toFixed(2)}% confidence).`}
          </Alert>
        )}
      </div>

      <div className="bg-white shadow-md rounded-xl p-8 w-full max-w-2xl space-y-6 mt-6">
        <div className="text-lg font-medium">Live Prediction (Tomorrow)</div>
        <div className="text-sm opacity-70 mb-4">
          Predict tomorrow's movement using today's live news and price data. 
          News is automatically fetched from Finnhub API.
        </div>

        <TextField
          label="Ticker symbol"
          value={tickerTomorrow}
          onChange={(e) => setTickerTomorrow(e.target.value.toUpperCase())}
          inputProps={{ maxLength: 8 }}
          fullWidth
        />

        <div className="flex items-center gap-4">
          <Button
            variant="contained"
            disabled={!canSubmitTomorrow || loadingTomorrow}
            sx={{ textTransform: 'none', borderRadius: '8px' }}
            onClick={handlePredictTomorrow}
          >
            {loadingTomorrow ? <CircularProgress size={24} color="inherit" /> : 'Predict Tomorrow'}
          </Button>
        </div>

        {errorTomorrow && <Alert severity="error">{errorTomorrow}</Alert>}

        {resultTomorrow && (
          <Alert severity={resultTomorrow.direction === 'UP' ? 'success' : 'warning'}>
            <div className="space-y-1">
              <div className="font-medium">
                Prediction for {resultTomorrow.date}:
              </div>
              <div>
                {resultTomorrow.direction === 'UP'
                  ? `The model predicts ${resultTomorrow.ticker} will go UP (${(resultTomorrow.probability * 100).toFixed(2)}% confidence).`
                  : `The model predicts ${resultTomorrow.ticker} will go DOWN (${(resultTomorrow.probability * 100).toFixed(2)}% confidence).`}
              </div>
              <div className="text-xs opacity-75 mt-2">
                Based on {resultTomorrow.headlines_used} headline{resultTomorrow.headlines_used !== 1 ? 's' : ''} from today
              </div>
            </div>
          </Alert>
        )}
      </div>
    </div>
  );
}

export default App;
