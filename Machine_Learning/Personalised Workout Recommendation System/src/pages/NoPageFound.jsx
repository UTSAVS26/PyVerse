import { Box } from "@mui/material";
import React from "react";

import { Navbar } from "../components/Navbar";

export const NoPageFound = () => {
  return (
    <>
      <Navbar showButton={true} />
      <Box
        height={"90vh"}
        sx={{ display: "flex", justifyContent: "center", alignItems: "center" }}
      >
        <h2>No Page Found...</h2>
      </Box>
    </>
  );
};
