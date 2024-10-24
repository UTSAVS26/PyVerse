import styled from "styled-components";

export const Container = styled.div`
  display: flex;
  align-items: center;
  justify-content: space-between;

  width: 1062px;
  margin-top: 23px;
  padding: 10px;

  background-color: #fff;
  border-radius: 100px;
  box-shadow: 0px 10px 20px 0px rgba(0, 0, 0, 0.1);

  .coingecko_logo {
    width: 150px;
    height: 37.5px;
  }

  @media (max-width: 1200px) {
    width: calc(100% - 32px);
    margin-left: 16px;
    margin-right: 16px;
  }

  @media (max-width: 480px) {
    flex-direction: column;
    gap: 16px;
    border-radius: 32px;
  }
`;

export const NavItems = styled.ul`
  display: flex;
  align-items: center;

  margin-bottom: 0;
  margin-top: 0;
  margin-right: 30px;
  gap: 32px;

  list-style: none;
`;

export const NavItem = styled.li`
  display: flex;
  align-items: center;

  gap: 12px;

  cursor: pointer;
  font-size: 18px;
  font-style: normal;
  font-weight: 500;
  line-height: 150%;

  a {
    text-decoration: none;
    color: #000;
  }

  &.active {
    a {
      background: linear-gradient(95deg, #ffb201 1.6%, #ff7a01 105.01%);
      background-clip: text;
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }
  }
`;
