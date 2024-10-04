import React from "react";

import "./MenuItem.css";

const MenuItem = ({ title, price, tags }) => (
  <div className="app__menuItem">
    <div className="app__menuItem-head">
      <div className="app__menuItem-name">
        <p className="p__cormorant" style={{ color: "#DCCA87" }}>
          {title}
        </p>
      </div>

      <div className="app__menuItem-dash" />

      <div className="app__menuItem-price">
        <p className="p__cormorant">{price}</p>
      </div>
    </div>

    <div className="app__menuItem-sub">
      <div className="p__opensans" style={{ color: "#AAA" }}>
        {tags}
      </div>
    </div>
  </div>
);

export default MenuItem;
