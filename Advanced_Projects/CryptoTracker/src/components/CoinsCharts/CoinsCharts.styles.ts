import styled from "styled-components";

export const ChartContainer = styled.div`
  width: 1200px;
  height: 600px;
  margin: 0 auto;
  padding: 20px;

  @media (max-width: 1280px) {
    width: 100%;
  }
`;

export const SelectsContainer = styled.div`
  display: flex;
  gap: 32px;
`;
