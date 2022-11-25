import React from 'react';
import {
  BrowserRouter, Routes, Route,
  // Outlet,
  // Navigate,
  // useLocation,
} from 'react-router-dom';
import EverytimeScreen from './screens/everytime_screen.js';
import GptGenerateScreen from './screens/gpt_generate_screen.js';

export default function AppRouter() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path='/' element={<EverytimeScreen/>}/>
        <Route path='/gptGenerate' element={<GptGenerateScreen/>}/>
      </Routes>
    </BrowserRouter>
  );
}