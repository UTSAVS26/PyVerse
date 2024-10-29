import axios from "axios";

const OPENAI_API_KEY = import.meta.env.VITE_OPENAI_API_KEY;

const getPrompt = (userInput) => {
  return `Generate a workout plan for a ${userInput.gender} who is ${userInput.age} years old and has a ${userInput.fitnessLevel} fitness level. They want to target ${userInput.targetMuscles} and have a ${userInput.workoutDuration}-minute workout.`;
};

export const generateWorkoutPlan = async (payload) => {
  const url = "https://api.openai.com/v1/completions";
  const headers = {
    "Content-Type": "application/json;charset=UTF-8",
    Charset: "utf-8",
    Authorization: `Bearer ${OPENAI_API_KEY}`,
  };

  const prompt = getPrompt(payload);
  const { data } = await axios.post(
    url,
    {
      model: "gpt-3.5-turbo-instruct",
      prompt,
      max_tokens: 150,
      temperature: 0,
    },
    { headers }
  );

  return { data };
};
