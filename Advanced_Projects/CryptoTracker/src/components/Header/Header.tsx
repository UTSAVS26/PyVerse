import { useCallback, useState } from "react";
import { Link, useLocation } from "react-router-dom";
import { Snackbar } from "@mui/material";
import { amber } from "@mui/material/colors";

import { ReactComponent as CoingeckoLogo } from "../../assets/coingecko.svg";

import { Container, NavItem, NavItems } from "./Header.styles";

export const Header = () => {
  const location = useLocation();

  const [open, setOpen] = useState(false);

  const isActivePath = useCallback(
    (path: string) => location.pathname === path,
    [location.pathname]
  );

  const handleChartsClick = (e: React.MouseEvent<HTMLAnchorElement>) => {
    const coinsSelected = localStorage.getItem("selected_coins_ids");

    if (!coinsSelected || JSON.parse(coinsSelected).length === 0) {
      e.preventDefault();
      setOpen(true);
    }
  };

  return (
    <Container>
      <Link to="/">
        <CoingeckoLogo className="coingecko_logo" />
      </Link>

      <NavItems>
        <NavItem className={isActivePath("/coins") ? "active" : ""}>
          <Link to="/coins">Coins List</Link>
        </NavItem>
        <NavItem className={isActivePath("/charts") ? "active" : ""}>
          <Link to="/charts" onClick={handleChartsClick}>
            Coins Charts
          </Link>
        </NavItem>
      </NavItems>

      <Snackbar
        open={open}
        autoHideDuration={3000}
        onClose={() => setOpen(false)}
        anchorOrigin={{ vertical: "bottom", horizontal: "right" }}
        message="Select at least one coin."
        ContentProps={{
          style: { backgroundColor: amber[700], minWidth: "200px" },
        }}
      />
    </Container>
  );
};
