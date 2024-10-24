import { useEffect, useMemo, useState } from "react";
import { DataGrid, GridRowSelectionModel } from "@mui/x-data-grid";
import { CircularProgress, Snackbar, TextField } from "@mui/material";
import { red } from "@mui/material/colors";

import { getCoins } from "src/api/coingecko";

import { coinsColumns } from "./columns";
import { coinsFilter } from "src/utils/functions";
import { Coin } from "@types";

import { BrandText, Container } from "./CoinsTable.styles";

export const CoinsTable = () => {
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<false | string>(false);

  const [coinsData, setCoinsData] = useState<Coin[]>([]);
  const [selectedCoins, setSelectedCoins] = useState<string[]>([]);

  const [searchQuery, setSearchQuery] = useState<string>("");
  const [pagination, setPagination] = useState({ page: 0, pageSize: 5 });

  const handleRowSelected = (params: GridRowSelectionModel) => {
    setSelectedCoins(params.map(String));
  };

  const handleSearchChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setSearchQuery(event.target.value);
  };

  useEffect(() => {
    const storedCoins = localStorage.getItem("selectedCoins");
    if (storedCoins) {
      setSelectedCoins(JSON.parse(storedCoins));
    }
  }, []);

  useEffect(() => {
    const selectedCoinsIdsString = localStorage.getItem("selected_coins_ids");
    if (selectedCoinsIdsString)
      setSelectedCoins(JSON.parse(selectedCoinsIdsString));

    const fetchData = async () => {
      try {
        const response = await getCoins();
        setCoinsData(response.data);
        setLoading(false);
      } catch (error) {
        console.error("Error fetching data:", error);
        if (error instanceof Error) setError(error.message);
        setLoading(false);
      }
    };

    fetchData();

    const intervalId = setInterval(fetchData, 15000);

    return () => clearInterval(intervalId);
  }, []);

  useEffect(() => {
    if (selectedCoins)
      localStorage.setItem("selected_coins_ids", JSON.stringify(selectedCoins));
  }, [selectedCoins]);

  const filteredCoins = useMemo(() => {
    return coinsFilter(coinsData, searchQuery);
  }, [searchQuery, coinsData]);

  if (loading) return <CircularProgress />;

  return (
    <Container>
      <BrandText>
        Powered by{" "}
        <a href="https://www.coingecko.com/" target="_blank" rel="noreferrer">
          CoinGecko API
        </a>
      </BrandText>

      <TextField
        label="Search by name or symbol"
        variant="outlined"
        className="search_input"
        value={searchQuery}
        onChange={handleSearchChange}
      />

      <DataGrid
        rows={filteredCoins}
        columns={coinsColumns}
        initialState={{
          pagination: {
            paginationModel: pagination,
          },
        }}
        onPaginationModelChange={(params) => {
          setPagination(params);
        }}
        rowSelectionModel={selectedCoins}
        onRowSelectionModelChange={handleRowSelected}
        pageSizeOptions={[5, 10, 25, 50, 100]}
        checkboxSelection
      />

      <Snackbar
        open={!!error}
        autoHideDuration={3000}
        onClose={() => setError(false)}
        anchorOrigin={{ vertical: "bottom", horizontal: "right" }}
        message={`API error: ${error}`}
        ContentProps={{
          style: { backgroundColor: red[500] },
        }}
      />
    </Container>
  );
};
