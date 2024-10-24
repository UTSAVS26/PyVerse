import React, { useEffect, useState } from "react";
import { Line } from "react-chartjs-2";
import axios from "axios";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from "chart.js";
import { ChartContainer, SelectsContainer } from "./CoinsCharts.styles";
import { FormControl, InputLabel, MenuItem, Select } from "@mui/material";
import { stringCapitalize } from "src/utils/functions";
import { ChartDataType, MarketChartDataType } from "@types";

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

export const CoinsCharts = () => {
  const [chartData, setChartData] = useState<ChartDataType | null>(null);

  const availableCoins = localStorage.getItem("selected_coins_ids");

  const [selectedCoin, setSelectedCoin] = useState(
    availableCoins && JSON.parse(availableCoins)[0]
  );

  const [days, setDays] = useState(30);

  const options = {
    responsive: true,
    plugins: {
      legend: {
        position: "top" as const,
      },
    },
  };

  useEffect(() => {
    const fetchChartData = async () => {
      try {
        const { data } = await axios.get<MarketChartDataType>(
          `https://api.coingecko.com/api/v3/coins/${selectedCoin}/market_chart?vs_currency=usd&days=${days}`
        );

        const dates = data.prices.map((pricePoint) =>
          new Date(pricePoint[0]).toLocaleDateString()
        );
        const prices = data.prices.map((pricePoint) => pricePoint[1]);

        const dataForChart: ChartDataType = {
          labels: dates,
          datasets: [
            {
              label: `${stringCapitalize(
                selectedCoin
              )} price in USD Last ${days} Days`,
              data: prices,
              borderColor: "rgba(75, 192, 192, 1)",
              fill: false,
            },
          ],
        };

        setChartData(dataForChart);
      } catch (error) {
        console.error("Error fetching chart data:", error);
      }
    };
    fetchChartData();
  }, [days, selectedCoin]);

  return (
    <ChartContainer>
      <SelectsContainer>
        <FormControl>
          <InputLabel id="demo-simple-select-standard-label">
            Time Period
          </InputLabel>
          <Select
            value={days}
            label="Time Period"
            onChange={(e) => setDays(Number(e.target.value))}
          >
            <MenuItem value={10}>10 days</MenuItem>

            <MenuItem value={30}>30 days</MenuItem>
            <MenuItem value={60}>60 days</MenuItem>
            <MenuItem value={180}>180 days</MenuItem>
            <MenuItem value={360}>360 days</MenuItem>
          </Select>
        </FormControl>

        <FormControl>
          <InputLabel id="demo-simple-select-standard-label">Coin</InputLabel>

          <Select
            value={selectedCoin}
            label="Coin"
            onChange={(e) => setSelectedCoin(e.target.value)}
          >
            {availableCoins &&
              JSON.parse(availableCoins).map((el: string) => (
                <MenuItem style={{ textTransform: "capitalize" }} value={el}>
                  {el}
                </MenuItem>
              ))}
          </Select>
        </FormControl>
      </SelectsContainer>

      {chartData && <Line data={chartData} options={options} />}
    </ChartContainer>
  );
};

export default CoinsCharts;
