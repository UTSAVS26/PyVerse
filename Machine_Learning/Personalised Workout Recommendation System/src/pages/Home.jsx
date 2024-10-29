import React from "react";
import { AppBar, Box, Container } from "@mui/material";

import { Navbar } from "../components/Navbar";
import { LandingHero } from "../components/LandingHero";

const Home = () => {
  return (
    <AppBar sx={{ background: "#000", height: "100vh" }}>
      <Navbar />
      <LandingHero />
    </AppBar>
  );
};

export default Home;
