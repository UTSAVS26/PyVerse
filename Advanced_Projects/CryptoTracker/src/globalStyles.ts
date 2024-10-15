import { createGlobalStyle } from "styled-components";

import { grey } from "@mui/material/colors";

export const GlobalStyle = createGlobalStyle`

   body {
    margin: 0;
    margin: 0;

    background-color: ${grey[100]};
    color: ${grey[900]};

    font-family: 'Roboto', sans-serif;
    
  }
`;
