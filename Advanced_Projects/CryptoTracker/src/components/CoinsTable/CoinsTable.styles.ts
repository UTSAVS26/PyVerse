import { grey } from "@mui/material/colors";
import styled from "styled-components";

export const Container = styled.div`
  display: flex;
  margin-top: 94px;

  position: relative;

  .search_input {
    position: absolute;
    top: -64px;
  }

  @media (max-width: 1280px) {
    width: calc(100% - 32px);
    margin-left: 16px;
    margin-right: 16px;
    margin-bottom: 40px;
    .MuiDataGrid-root {
      width: 100%;
    }
  }
`;

export const BrandText = styled.span`
  position: absolute;
  right: 0px;
  bottom: -24px;
  font-size: 12px;
  line-height: 120%;
  color: ${grey[700]};

  a {
    text-decoration: none;
  }
`;
