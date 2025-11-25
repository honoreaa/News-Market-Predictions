# Frontend

React + Vite frontend for visualizing historical ML model performance.

1. Install dependencies

       `cd frontend && npm install`

2. Make sure the backend is running (see main README.md)

3. Start the frontend dev server

       `npm run dev`

The frontend will be available at `http://localhost:5173`. It connects to the backend API at `http://localhost:8000` by default. You can change this by setting `VITE_API_BASE_URL` in a `.env` file in the `frontend/` directory.

The frontend provides a historical performance dashboard that shows:
- Overall model performance metrics (all predictions vs predictions with mapped news headlines)
- Per-ticker performance summary table with accuracy and prediction counts
- Detailed view for any ticker showing all historical predictions
- Headline search to find predictions by news text

Note: This dashboard analyzes historical performance of the next-day prediction model across the dataset (2018-2023), distinguishing between predictions that had mapped news headlines versus those that didn't.
