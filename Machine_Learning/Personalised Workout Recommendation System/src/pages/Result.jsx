import React from "react";
import ReactMarkdown from "react-markdown";
import { Box, Container } from "@mui/material";

import { Navbar } from "../components/Navbar";
import { useSelector } from "react-redux";

const Result = () => {
  const { data } = useSelector((state) => state.workout);

  return (
    <>
      <Navbar showButton={true} />
      <Container
        maxWidth="xl"
        sx={{
          height: "90vh",
          display: "flex",
          justifyContent: "center",
          overflowX: "auto",
          background: "#000",
        }}
      >
        <Box
          width="80%"
          fontSize={"20px"}
          color={"#fff"}
          fontWeight={"700"}
          lineHeight={"1.5em"}
        >
          {data?.text ? (
            <ReactMarkdown children={data.text} />
          ) : (
            <Box
              height={"80vh"}
              sx={{
                display: "flex",
                justifyContent: "center",
                alignItems: "center",
              }}
            >
              <h2>No workout plan found, please generate again...</h2>
            </Box>
          )}
        </Box>
      </Container>
    </>
  );
};

export default Result;