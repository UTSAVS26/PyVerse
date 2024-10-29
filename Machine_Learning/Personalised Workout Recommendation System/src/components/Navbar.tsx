import React from "react";
import {
  AppBar,
  Box,
  Button,
  CardMedia,
  Container,
  Toolbar,
} from "@mui/material";
import { useNavigate } from "react-router-dom";

import logo from "../assets/smartFitLogo.png";

export const Navbar = ({ showButton = "true" }) => {
  const navigate = useNavigate();
  return (
    <AppBar position="sticky" sx={{ background: "#000" }}>
      <Container maxWidth="xl" sx={{ padding: "20px" }}>
        <Toolbar sx={{ display: "flex", justifyContent: "space-between" }}>
          <Box onClick={() => navigate("/")}>
            <CardMedia
              component={"img"}
              image={logo}
              height={60}
              alt="logo"
              sx={{ cursor: "pointer" }}
            />
          </Box>
          {showButton && (
            <Box sx={{ display: "flex", alignItems: "center" }}>
              <Button
                variant="outlined"
                sx={{
                  borderRadius: "30px",
                  color: "#000",
                  background: "#fff",
                  padding: {
                    sm: "10px 16px",
                    md: "12px 20px",
                  },
                  fontWeight: "700",
                  textTransform: "none",
                  fontSize: "16px",
                  fontFamily: "inherit",
                  letterSpacing: {
                    md: ".1rem",
                  },
                  "&:hover": {
                    background: "#d7f7f4",
                  },
                }}
                onClick={() => navigate("/conversation")}
              >
                Generate workout
              </Button>
            </Box>
          )}
        </Toolbar>
      </Container>
    </AppBar>
  );
};
