import Box from '@mui/material/Box';
import InputLabel from '@mui/material/InputLabel';
import MenuItem from '@mui/material/MenuItem';
import FormControl from '@mui/material/FormControl';
import Select from '@mui/material/Select';
import Button from '@mui/material/Button';
import GitHubIcon from '@mui/icons-material/GitHub';

import { useState } from 'react';

function App() {
  const [model, setModel] = useState(1);

  const handleChange = (event) => {
    setModel(event.target.value);
  };

  return (
    <div className="flex flex-col items-center min-h-screen py-12 px-6 bg-[#F3F6FB]">

      {/* Header */}
      <div className="w-full max-w-3xl mb-10">
        <div className="text-4xl font-semibold">ECS171: News Market Predictor</div>
        <Button
          variant="contained"
          startIcon={<GitHubIcon />}
          href="https://github.com/honoreaa/News-Market-Predictions"
          target="_blank"
          sx={{ textTransform: "none", borderRadius: "8px" }}
        >
          Project Github!
        </Button>
        <div className="text-sm mt-1 leading-relaxed opacity-80">
          Group 4: Honore Alexander, Owen Holt, Pranavi Khanna, Dylan Lim,
          Yihong Li, Ethan Lee, Dan Firstenberg, Hyeongseung Nam, Oscar Pineda,
          Kevin Zhang, Zachary Chan, Vicente Aguayo
        </div>

      </div>

      {/* Model Select */}
      <div className="bg-white shadow-md rounded-xl p-8 w-full max-w-sm mb-8">
        <div className="text-lg font-medium mb-4 text-center">
          Select Prediction Model
        </div>

        <Box>
          <FormControl fullWidth>
            <InputLabel id="demo-simple-select-label">Model</InputLabel>
            <Select
              labelId="demo-simple-select-label"
              id="demo-simple-select"
              value={model}
              label="Model"
              onChange={handleChange}
            >
              <MenuItem value={1}>Model 1</MenuItem>
              <MenuItem value={2}>Model 2</MenuItem>
              <MenuItem value={3}>Model 3</MenuItem>
            </Select>
          </FormControl>
        </Box>
      </div>

      {/* Output Area */}
      <div className="bg-white shadow-md rounded-xl p-8 w-full max-w-2xl">
        <div className="text-lg font-medium mb-4">
          Output
        </div>

        <div className="text-gray-700 opacity-90 leading-relaxed">
          output appear here
        </div>
      </div>

    </div>
  )
};

export default App;
