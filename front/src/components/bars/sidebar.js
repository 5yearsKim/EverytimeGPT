import React from 'react';
import {Home as HomeIcon, } from '@mui/icons-material';
import {IconButton} from '@mui/material';
import { useNavigate } from 'react-router-dom';

import 'css/navbar.css';

export default function Sidebar() {
  const navigate = useNavigate();

  return (
    <div id='sidebar'>
      <IconButton
        onClick={() => navigate('/')}
      >
        <HomeIcon className='icon'/>
      </IconButton>
    </div>
  );
}