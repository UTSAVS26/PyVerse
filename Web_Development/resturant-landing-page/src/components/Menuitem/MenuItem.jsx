import React from "react";
import "./MenuItem.css";

// MENU ITEM COMPONENT
// Receives title, price, and tags as props
const MenuItem = ({ title, price, tags }) => (
  <div className="app__menuItem">

    {/* Menu item header containing the name and price */}
    <div className="app__menuItem-head">
      <div className="app__menuItem-name">
        <p className="p__cormorant" style={{ color: "#DCCA87" }}>
          {title}
        </p>
      </div>

      <div className="app__menuItem-dash" />

      {/* Displaying the item price */}
      <div className="app__menuItem-price">
        <p className="p__cormorant">{price}</p>
      </div>
    </div>

    {/* Displaying the item tags */}
    <div className="app__menuItem-sub">
      <div className="p__opensans" style={{ color: "#AAA" }}>
        {tags}
      </div>
    </div>
  </div>
);

export default MenuItem;
