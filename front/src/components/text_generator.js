import React, {useState} from 'react';
import {TextField, Button, CircularProgress, IconButton} from '@mui/material';
import {Close as CloseIcon} from '@mui/icons-material';
import {generateText} from 'networks/gpt_generate.js';

import 'css/gpt_generate.css';

export default function TextGenerator({initialText='사실 어제 나 ', showKeywords=true}) {
  const [query, setQuery] = useState(initialText);
  const [results, setResults] = useState([]);
  const [newKw, setNewKw] = useState('');
  const [isLoading, setIsLoading] = useState(false);


  const onSubmit = async () => {
    if (!query) {
      return;
    }
    const keywords = newKw.split();
    try {
      setIsLoading(true);
      const generated =  await generateText(query, keywords);
      setResults(generated);
    } catch (e) {
      console.warn(e);
    } finally {
      setIsLoading(false);
    }
  };


  const renderKeywords = () => {
    if (showKeywords == false) {
      return null;
    }
    return (
      <div className='keyword-container'>
        <span className='title' >키워드:</span>

        <div style={{width: '20px'}}/>

        <TextField
          value={newKw}
          onChange={(e) => setNewKw(e.target.value)}
          variant='standard'
          placeholder='띄어쓰기로 구분해서 입력'
          InputProps={{
            endAdornment: <IconButton
              onClick={() => setNewKw('')}
            >
              <CloseIcon/>
            </IconButton>,
          }}
        />
      </div>
    );
  };

  const renderGenerated = (text) => {
    text = text.trim();
    const rest = text.split(query);
    return <span><span className='bold'>{query}</span>{rest[1]}</span>;
  };

  const renderLoadingIndicator = () => {
    if (!isLoading) {
      return null;
    }
    return <CircularProgress wid className='circular-progress'/>;
  };

  return (
    <div id='text-generator'>
      {renderLoadingIndicator()}
      {renderKeywords()}

      <TextField
        variant='outlined'
        value={query}
        onChange={(e) => {
          setQuery(e.target.value);
          setResults([]);
        }}
        multiline
        fullWidth
        rows={4}
      />

      <div style={{padding: '2px'}}/>

      <div className='button-row'>
        <Button
          variant='contained'
          onClick={onSubmit}
          disabled={isLoading}
        >
          문장 완성하기
        </Button>
      </div>

      <div style={{padding: '10px'}}/>

      <div className='result-paper'>
        {results.map((result, idx) => {
          return (
            <div key={idx} className='result-box'>
              {renderGenerated(result)}
            </div>
          );
        })}

      </div>
    </div>
  );
}
