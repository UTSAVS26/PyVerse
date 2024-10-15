import React from "react";
import { createBrowserRouter, Outlet, RouterProvider } from "react-router-dom";

import { CoinsTable } from "./components/CoinsTable/CoinsTable";
import { CoinsCharts } from "./components/CoinsCharts/CoinsCharts";
import { Header } from "./components/Header/Header";

import { Wrapper } from "./styles";

const router = createBrowserRouter([
  {
    path: "/",
    element: (
      <>
        <Header />
        <Outlet />
      </>
    ),

    children: [
      { path: "coins", element: <CoinsTable /> },
      { path: "charts", element: <CoinsCharts /> },
    ],
  },
]);

export const App: React.FC = () => {
  return (
    <Wrapper>
      <RouterProvider router={router} />
    </Wrapper>
  );
};
