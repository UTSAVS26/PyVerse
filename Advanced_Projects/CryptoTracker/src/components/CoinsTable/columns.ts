export const coinsColumns = [
  { field: "id", headerName: "ID", width: 250 },
  { field: "symbol", headerName: "Symbol", width: 150 },
  { field: "name", headerName: "Name", width: 250 },
  {
    field: "current_price",
    headerName: "Price (USD)",
    width: 200,
    type: "number",
    valueFormatter: ({ value }: { value: number }) => `$${value}`,
  },
  {
    field: "market_cap",
    headerName: "Market Cap",
    width: 200,
    type: "number",
    valueFormatter: ({ value }: { value: number }) =>
      `$${value.toLocaleString()}`,
  },
  {
    field: "market_cap_rank",
    headerName: "Rank",
    width: 100,
    type: "number",
  },
];
