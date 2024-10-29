import { createAsyncThunk, createSlice } from "@reduxjs/toolkit";

import { generateWorkoutPlan } from "./workoutApi";

const initialState = {
  loading: false,
  data: {},
  error: null,
};

export const generateWorkout = createAsyncThunk(
  "workout/generate",
  async (payload) => {
    console.log("payload-->", payload);
    try {
      const response = await generateWorkoutPlan(payload);
      console.log("🚀 ~ file: workoutSlice.js:17 ~ response:", response);
      return response;
    } catch (error) {
      throw new Error(error.message);
    }
  }
);

const workoutSlice = createSlice({
  name: "workout",
  initialState,
  reducers: {},
  extraReducers: (builder) => {
    builder
      .addCase(generateWorkout.pending, (state) => {
        state.loading = true;
      })
      .addCase(generateWorkout.fulfilled, (state, action) => {
        state.loading = false;
        state.data = action.payload?.data?.choices[0];
        state.error = null;
      })
      .addCase(generateWorkout.rejected, (state, action) => {
        state.loading = false;
        state.data = null;
        state.error = action.error.message;
      });
  },
});

export const workoutReducer = workoutSlice.reducer;
