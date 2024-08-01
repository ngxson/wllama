// import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App.tsx';
import './index.css';

ReactDOM.createRoot(document.getElementById('root')!).render(
  // TODO: we disable strict mode because some dispatchers are fired twice
  //<React.StrictMode>
  <App />
  //</React.StrictMode>,
);
