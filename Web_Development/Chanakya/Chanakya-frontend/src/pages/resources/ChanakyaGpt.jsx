import { useState } from "react";
import Groq from "groq-sdk";
import { ClipLoader } from "react-spinners";

const groq = new Groq({
  apiKey: "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",
  dangerouslyAllowBrowser: true,
});

const ChanakyaGpt = () => {
  const [prompt, setPrompt] = useState();
  const [response, setResponse] = useState();
  const [loader, setLoader] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    console.log(prompt);
    //making an api call to groq
    try {
      setLoader(true);
      const response = await groq.chat.completions.create({
        messages: [
          {
            role: "user",
            content: `
                Answer this question as if you are chanakya and the question is :
                    ${prompt}
                `,
          },
        ],
        model: "llama3-8b-8192",
      });
      setLoader(false);
      console.log(response.choices[0]?.message?.content || "");
      setResponse(response.choices[0]?.message?.content);
    } catch (error) {
      console.log("Error occured");
    }
    setPrompt("");
  };

  return (
    <div className="font-bold flex w-screen items-center rounded-lg m-2">
      <form onSubmit={(e) => handleSubmit(e)} className="shadow-md p-10">
        <div className="flex flex-col gap-7">
          <h1 className="text-3xl font-semibold">Ask your questions</h1>
          <div className="flex flex-col justify-center items-center">
            <input
              onChange={(e) => setPrompt(e.target.value)}
              value={prompt}
              type="text"
              placeholder="type something..."
              className="w-screen text-xl font-normal border-gray-300 rounded-lg border-2 p-2"
            />
            {loader ? (
              <button className="text-white w-full bg-white mt-2 p-2 rounded-md cursor-pointer">
                <ClipLoader />
              </button>
            ) : (
              <button className="text-white w-full bg-black mt-2 p-2 rounded-md cursor-pointer">
                Search
              </button>
            )}
          </div>
        </div>
        <div className="font-normal mt-2">{response && <>{response}</>}</div>
      </form>
    </div>
  );
};

export default ChanakyaGpt;
