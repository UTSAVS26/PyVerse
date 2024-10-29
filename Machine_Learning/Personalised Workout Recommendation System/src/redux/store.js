import { configureStore } from "@reduxjs/toolkit";

import { workoutReducer } from "./workoutSlice";

export const store = configureStore({
  reducer: {
    workout: workoutReducer,
  },
});
