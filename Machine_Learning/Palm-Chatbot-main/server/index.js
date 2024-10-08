const express = require("express");
const cors = require("cors");
require("dotenv").config();

const { DiscussServiceClient } = require("@google-ai/generativelanguage");
const { GoogleAuth } = require("google-auth-library");

const app = express();
const port = 3000;
const MODEL_NAME = "models/chat-bison-001";
const API_KEY = process.env.API_KEY;
const client = new DiscussServiceClient({
  authClient: new GoogleAuth().fromAPIKey(API_KEY),
});

const CONTEXT =
  "Respond to all questions with atleast 300 words";
const EXAMPLES = [
  {
    input: { content: "What is the capital of California?" },
    output: {
      content: `If the capital of California is what you seek,
Sacramento is where you ought to peek.`,
    },
  },
];

let messages = [];

app.use(express.json());
app.use(
  cors({
    origin: "*",
  })
);

app.post("/api/chatbot", async (req, res) => {
  const requestData = req.body;

  if (requestData && requestData.message) {
    const message = requestData.message;
    messages.push({ content: message });

    const result = await client.generateMessage({
      model: MODEL_NAME,
      prompt: {
        context: CONTEXT,
        examples: EXAMPLES,
        messages,
      },
    });

    const messageResult = result[0].candidates[0].content;
    messages.push({ content: messageResult });

    res.json({ message: messageResult, agent: "chatbot" });
  } else {
    res.status(400).json({ error: "Content not provided" });
  }
});

app.listen(port, () => {
  console.log(`Server is running on port ${port}`);
});
