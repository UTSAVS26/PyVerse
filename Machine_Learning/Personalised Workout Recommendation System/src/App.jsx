import { Route, Routes } from "react-router-dom";
import { Suspense, lazy } from "react";

const Home = lazy(() => import("./pages/Home"));
const Conversation = lazy(() => import("./pages/Conversation"));
const Result = lazy(() => import("./pages/Result"));

import { NoPageFound } from "./pages/NoPageFound";

function App() {
  return (
    <>
      <Suspense follback={<div>Loading...</div>}>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/conversation" element={<Conversation />} />
          <Route path="/result" element={<Result />} />
          <Route path="*" element={<NoPageFound />} />
        </Routes>
      </Suspense>
    </>
  );
}

export default App;
