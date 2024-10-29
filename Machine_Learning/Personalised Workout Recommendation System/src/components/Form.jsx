import {
  Box,
  Button,
  CircularProgress,
  Container,
  FormControl,
  InputLabel,
  MenuItem,
  Select,
  TextField,
  Typography,
} from "@mui/material";
import React, { useState } from "react";
import { useDispatch, useSelector } from "react-redux";
import { useNavigate } from "react-router-dom";

import { generateWorkout } from "../redux/workoutSlice";

export const Form = () => {
  const [formData, setFormData] = useState({
    age: null,
    gender: null,
    fitnessLevel: null,
    targetMuscles: null,
    workoutDuration: null,
  });
  const dispatch = useDispatch();
  const navigate = useNavigate();

  const { loading, error } = useSelector((state) => state.workout);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData({
      ...formData,
      [name]: value,
    });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    await dispatch(generateWorkout(formData));
    navigate("/result");
  };

  return (
    <Container maxWidth="xl" sx={{ marginTop: "2%" }}>
      <Typography
        variant="h4"
        component="h4"
        fontFamily={"inherit"}
        sx={{
          fontSize: {
            sm: "3rem",
            md: "2rem",
            lg: "2.25rem",
          },
        }}
      >
        Fill below form to generate your personalized workout plan
      </Typography>
      <Box
        sx={{
          marginTop: "10px",
          display: "flex",
          flexWrap: "wrap",
        }}
        component={"form"}
        gap={"20px"}
        onSubmit={handleSubmit}
      >
        <FormControl fullWidth required>
          <TextField
            variant="outlined"
            label="Age"
            sx={{ borderRadius: "10px" }}
            name="age"
            onChange={handleChange}
            required
          />
        </FormControl>
        <FormControl fullWidth required>
          <InputLabel id="gender">Gender</InputLabel>
          <Select
            labelId="gender"
            label="Gender"
            onChange={handleChange}
            name="gender"
            required
            defaultValue={""}
          >
            <MenuItem value={"male"}>Male</MenuItem>
            <MenuItem value={"female"}>Femal</MenuItem>
          </Select>
        </FormControl>
        <FormControl fullWidth required>
          <InputLabel id="fitnessLevel">Fitness Level</InputLabel>
          <Select
            labelId="fitnessLevel"
            label="Fitness Level"
            onChange={handleChange}
            name="fitnessLevel"
            required
            defaultValue={""}
          >
            <MenuItem value={"beginner"}>Beginner</MenuItem>
            <MenuItem value={"intermediate"}>Intermediate</MenuItem>
            <MenuItem value={"advanced"}>Advanced</MenuItem>
            <MenuItem value={"athlete"}>Athlete</MenuItem>
          </Select>
        </FormControl>
        <FormControl fullWidth required>
          <InputLabel id="muscleGroup">Target Muscle Group</InputLabel>
          <Select
            labelId="muscleGroup"
            label="Target Muscle Group"
            onChange={handleChange}
            name="targetMuscles"
            required
            defaultValue={""}
          >
            <MenuItem value={"full body"}>Full Body</MenuItem>
            <MenuItem value={"upper body"}>Upper Body</MenuItem>
            <MenuItem value={"lower body"}>Lower Body</MenuItem>
            <MenuItem value={"core"}>Core</MenuItem>
            <MenuItem value={"chest"}>Chest</MenuItem>
            <MenuItem value={"back"}>Back</MenuItem>
            <MenuItem value={"shoulders"}>Shoulders</MenuItem>
            <MenuItem value={"legs"}>Legs</MenuItem>
            <MenuItem value={"arms"}>Arms</MenuItem>
          </Select>
        </FormControl>
        <FormControl fullWidth required>
          <InputLabel id="workoutDuration">Desired Workout Duration</InputLabel>
          <Select
            labelId="workoutDuration"
            label="Desired Workout Duration"
            onChange={handleChange}
            name="workoutDuration"
            required
            defaultValue={""}
          >
            <MenuItem value={"15"}>15 minutes</MenuItem>
            <MenuItem value={"30"}>30 minutes</MenuItem>
            <MenuItem value={"45"}>45 minutes</MenuItem>
            <MenuItem value={"60"}>60 minutes</MenuItem>
            <MenuItem value={"75"}>75 minutes</MenuItem>
            <MenuItem value={"90"}>90 minutes</MenuItem>
          </Select>
        </FormControl>
        <FormControl fullWidth>
          <Button
            variant="contained"
            sx={{ padding: "14px", borderRadius: "5px" }}
            type="submit"
            disabled={loading}
          >
            {loading ? (
              <CircularProgress color="inherit" size={16} />
            ) : (
              "Generate Workout"
            )}
          </Button>
        </FormControl>
      </Box>
    </Container>
  );
};
