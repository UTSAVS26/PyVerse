import React, { createContext, useState, useEffect } from 'react';

export const Context = createContext();

const ContextProvider = ({ children }) => {
  const getInitialDarkMode = () => localStorage.getItem('darkMode') === 'true';

  const [progress, setProgress] = useState(0);
  const [isDarkMode, setDarkMode] = useState(getInitialDarkMode);

  useEffect(() => {
    localStorage.setItem('darkMode', isDarkMode);
  }, [isDarkMode]);

  const toggleTheme = () => {
    setDarkMode((prevMode) => {
      const newMode = !prevMode;
      localStorage.setItem('darkMode', newMode ? 'dark' : 'light');
      return newMode;
    });
  };

  return (
    <Context.Provider value={{
      progress, setProgress,
      isDarkMode, toggleTheme,
    }}>
      {children}
    </Context.Provider>
  );
};

export default ContextProvider;
