import React from "react";
import { Box, Button, Container, Typography } from "@mui/material";
import TypewriterComponent from "typewriter-effect";
import { useNavigate } from "react-router-dom";

export const LandingHero = () => {
  const navigate = useNavigate();
  return (
    <Container
      maxWidth="xl"
      sx={{
        height: "80vh",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        overflowX: "auto",
      }}
    >
      <Box
        sx={{
          maxWidth: {
            sm: "80%",
            md: "60%",
          },
          margin: "auto",
          zIndex: 10,
          color: "white",
        }}
      >
        <Typography
          variant="h1"
          sx={{
            fontFamily: "inherit",
            fontWeight: 900,
            fontSize: {
              xs: "2.5rem",
              sm: "3rem",
              md: "6rem",
            },
          }}
        >
          The Best AI Tool for
        </Typography>
        <Typography
          variant="h1"
          sx={{
            color: "#ed40bc",
            fontSize: {
              xs: "2.5rem",
              sm: "3rem",
              md: "6rem",
            },
          }}
        >
          <TypewriterComponent
            options={{
              autoStart: true,
              loop: true,
              strings: [
                "Look fit",
                "Personalized Workout Plan",
                "Make you sweat",
              ],
            }}
          />
        </Typography>
        <Box sx={{ display: "flex", justifyContent: "center" }}>
          <Button
            variant="outlined"
            sx={{
              borderRadius: "30px",
              color: "#000",
              background: "#fff",
              padding: "12px 20px",
              fontWeight: "700",
              textTransform: "none",
              fontSize: "16px",
              fontFamily: "inherit",
              letterSpacing: ".1em",
              marginTop: "20px",
              "&:hover": {
                background: "#d7f7f4",
              },
            }}
            onClick={() => navigate("/conversation")}
          >
            Start Generation for Free
          </Button>
        </Box>
      </Box>
    </Container>
  );
};
