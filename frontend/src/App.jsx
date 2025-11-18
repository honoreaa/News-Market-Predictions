import Box from '@mui/material/Box';
import InputLabel from '@mui/material/InputLabel';
import MenuItem from '@mui/material/MenuItem';
import FormControl from '@mui/material/FormControl';
import Select from '@mui/material/Select';

import { useState } from 'react';

function App() {
  const [model, setModel] = useState(1);

  const handleChange = (event) => {
    setModel(event.target.value);
  };

  return (
    <div className='flex flex-col justify-center items-center'>
      <div className='header w-[50rem]'>
        <div className='text-3xl text-left'>ECS171: News Market Predictor</div>
        <div className='text-sm'>Group 4: Honore Alexander, Owen Holt, Pranavi Khanna,  Dylan Lim, Yihong Li, Ethan Lee, Dan Firstenberg, Hyeongseung Nam, Oscar Pineda, Kevin Zhang, Zachary Chan, Vicente Aguayo
        </div>
      </div>

      <div className='Model Selector mt-[5rem]'>
        <Box sx={{ minWidth: 120 }}>
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
    </div>
  )
};

export default App;
