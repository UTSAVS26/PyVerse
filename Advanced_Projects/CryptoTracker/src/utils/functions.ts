import { Coin } from "@types";

export const stringCapitalize = (str: string) => {
  if (!str) return "";
  return str.charAt(0).toUpperCase() + str.slice(1).toLowerCase();
};

export const coinsFilter = (coinsData: Coin[], searchQuery: string) => {
  return coinsData.filter((coin) => {
    const nameMatch = coin.name
      .toLowerCase()
      .includes(searchQuery.toLowerCase());
    const symbolMatch = coin.symbol
      .toLowerCase()
      .includes(searchQuery.toLowerCase());
    return nameMatch || symbolMatch;
  });
};
